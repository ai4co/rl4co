import time
import os
import logging
import requests
import sys
import hashlib
import urllib
from tqdm.auto import tqdm
from typing import Optional, Any
import six
from six.moves import urllib_parse
import warnings
import textwrap
import re

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"

# Set logging level to INFO only for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_url(
    url: str,
    root: str,
    filename: Optional[str] = None,
    session: requests.Session = requests.Session(),
    key: Optional[str] = None,
    proxy: str = None,
    md5: Optional[str] = None,
    max_redirect_hops: int = 3,
    verify: Optional[bool] = None,
    timeout: int = 60,
    retry_max: int = 500,
    sleep_max: int = 120,
    chunk_size: int = 1024,
    show_progress: bool = True,
    logging_level: int = logging.INFO,
) -> None:
    """Download a file from a url and place it in root. Supports robust downloads with resuming,
    connection error handling, proxies, authentication, and more.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        session (requests.Session, optional): Session to use for HTTP requests.
        key (str, optional): authentication key in the form username:password
        proxy (str, optional): Proxy URL.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        max_redirect_hops (int, optional): Maximum number of redirects to follow
        verify (bool, optional): Whether to verify SSL certificates
        timeout (int, optional): Timeout for HTTP requests
        retry_max (int, optional): Maximum number of retries
        sleep_max (int, optional): Maximum number of seconds to sleep between retries
        chunk_size (int, optional): Number of bytes to read into memory at once
        show_progress (bool, optional): Whether to show a progress bar
        logging_level (int, optional): Logging level
    """
    # Set logging level
    logger.setLevel(logging_level)

    # Expand redirect hops if needed
    url = _get_redirect_url(url, max_hops=max_redirect_hops)

    # Additional session configuration
    if key:
        session.auth = tuple(key.split(":", 2))
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}

    # Try to get Google Drive url and if it exists, use it
    url_drive, filename_drive = get_url_filename_drive(url, session, verify)

    # Expand root and filename
    root = os.path.expanduser(root)
    if not filename and not url_drive:
        response = session.head(url, allow_redirects=True)
        content_disposition = response.headers.get("Content-Disposition")
        if content_disposition:
            filename = content_disposition.split(";")[-1]
        else:
            filename = os.path.basename(urllib.parse.urlparse(response.url).path)
    elif url_drive:
        url = url_drive
        filename = filename_drive if not filename else filename

    fpath = os.path.join(root, filename)
    os.makedirs(root, exist_ok=True)

    # Check if file is already present locally
    if md5 is not None and check_integrity(fpath, md5):
        logger.info("Using downloaded and verified file: " + fpath)
        return

    # Get file size
    try:
        downloaded = os.path.getsize(fpath)
    except FileNotFoundError:
        downloaded = 0
    size = int(session.get(url, stream=True).headers["Content-Length"])
    if downloaded == size:
        logger.info("File %s already downloaded", filename)
        return
    elif downloaded > size:
        raise RuntimeError("File %s is corrupted" % filename)

    logger.info("Downloading %s to %s (%s)", url, filename, format_bytes(size))

    mode = "ab"  # append to file if it exists
    sleep = 10
    tries = 0
    headers = {"Range": "bytes=%d-" % downloaded, "User-Agent": USER_AGENT}

    # Main loop: download the file in chunks with retries and resume on failure
    while tries < retry_max:
        r = robust_wrapper(session.get, retry_max, sleep_max)(
            url,
            stream=True,
            verify=verify,
            headers=headers,
            timeout=timeout,
        )
        try:
            r.raise_for_status()

            with tqdm(
                total=size,
                unit_scale=True,
                unit_divisor=1024,
                unit="B",
                initial=downloaded,
                disable=not show_progress,
            ) as pbar:
                with open(fpath, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))

        except requests.exceptions.ConnectionError as e:
            logger.error("Download interrupted: %s" % (e,))
        finally:
            r.close()

        if downloaded >= size:
            break

        logger.error(
            "Download incomplete, downloaded %s / %s"
            % (format_bytes(downloaded), format_bytes(size))
        )
        logger.warning("Sleeping %s seconds" % (sleep,))
        time.sleep(sleep)
        mode = "ab"

        downloaded = os.path.getsize(fpath)
        sleep *= 1.5
        if sleep > sleep_max:
            sleep = sleep_max
        headers = {"Range": "bytes=%d-" % downloaded, "User-Agent": USER_AGENT}
        tries += 1
        logger.warning("Resuming from downloaded %s" % (format_bytes(downloaded),))

    if downloaded != size:
        raise Exception(
            "Download failed: downloaded %s / %s"
            % (format_bytes(downloaded), format_bytes(size))
        )

    if md5 is not None:
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")
        else:
            logger.info("File integrity verified!")


def robust_wrapper(call, retry_max=500, sleep_max=120):
    def retriable(code, reason):
        if code in [
            requests.codes.internal_server_error,
            requests.codes.bad_gateway,
            requests.codes.service_unavailable,
            requests.codes.gateway_timeout,
            requests.codes.too_many_requests,
            requests.codes.request_timeout,
        ]:
            return True

        return False

    def wrapped(*args, **kwargs):
        tries = 0
        while tries < retry_max:
            try:
                r = call(*args, **kwargs)
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
            ) as e:
                r = None
                logger.warning(
                    "Recovering from connection error [%s], attempts %s of %s",
                    e,
                    tries,
                    retry_max,
                )

            if r is not None:
                if not retriable(r.status_code, r.reason):
                    return r
                try:
                    logger.warning(r.json()["reason"])
                except Exception:
                    pass
                logger.warning(
                    "Recovering from HTTP error [%s %s], attempts %s of %s",
                    r.status_code,
                    r.reason,
                    tries,
                    retry_max,
                )

            tries += 1

            logger.warning("Retrying in %s seconds", sleep_max)
            time.sleep(sleep_max)
            logger.info("Retrying now...")

    return wrapped


def _parse_gdrive_url(url, warning=True):
    """Parse URLs especially for Google Drive links.
    file_id: ID of file on Google Drive.
    is_download_link: Flag if it is download link of Google Drive.
    """
    parsed = urllib_parse.urlparse(url)
    query = urllib_parse.parse_qs(parsed.query)
    is_gdrive = parsed.hostname in ["drive.google.com", "docs.google.com"]
    is_download_link = parsed.path.endswith("/uc")

    if not is_gdrive:
        return None

    file_id = None
    if "id" in query:
        file_ids = query["id"]
        if len(file_ids) == 1:
            file_id = file_ids[0]
    else:
        patterns = [r"^/file/d/(.*?)/view$", r"^/presentation/d/(.*?)/edit$"]
        for pattern in patterns:
            match = re.match(pattern, parsed.path)
            if match:
                file_id = match.groups()[0]
                break

    if warning and not is_download_link:
        warnings.warn(
            "You specified a Google Drive link that is not the correct link "
            "to download a file. You might want to try `--fuzzy` option "
            "or the following url: {url}".format(
                url="https://drive.google.com/uc?id={}".format(file_id)
            )
        )

    return (
        "https://drive.google.com/uc?id={id}".format(id=file_id),
        file_id,
        is_download_link,
    )


def get_url_from_gdrive_confirmation(contents):
    url = ""
    for line in contents.splitlines():
        m = re.search(r'href="(\/uc\?export=download[^"]+)', line)
        if m:
            url = "https://docs.google.com" + m.groups()[0]
            url = url.replace("&amp;", "&")
            break
        m = re.search('id="download-form" action="(.+?)"', line)
        if m:
            url = m.groups()[0]
            url = url.replace("&amp;", "&")
            break
        m = re.search('"downloadUrl":"([^"]+)', line)
        if m:
            url = m.groups()[0]
            url = url.replace("\\u003d", "=")
            url = url.replace("\\u0026", "&")
            break
        m = re.search('<p class="uc-error-subcaption">(.*)</p>', line)
        if m:
            error = m.groups()[0]
            raise RuntimeError(error)
    if not url:
        raise RuntimeError(
            "Cannot retrieve the public link of the file. "
            "You may need to change the permission to "
            "'Anyone with the link', or have had many accesses."
        )
    return url


def indent(text, prefix):
    def prefixed_lines():
        for line in text.splitlines(True):
            yield (prefix + line if line.strip() else line)

    return "".join(prefixed_lines())


def get_url_filename_drive(url, sess, verify):
    url_origin = url
    url, gdrive_file_id, is_gdrive_download_link = _parse_gdrive_url(url)

    while True:
        try:
            res = sess.get(
                url, headers={"User-Agent": USER_AGENT}, stream=True, verify=verify
            )
        except requests.exceptions.ProxyError as e:
            logger.error(
                "An error has occurred using proxy:", sess.proxy, file=sys.stderr
            )
            logger.error(e, file=sys.stderr)
            return None, None

        if "Content-Disposition" in res.headers:
            # This is the file
            break
        if not (gdrive_file_id and is_gdrive_download_link):
            break

        # Need to redirect with confirmation
        try:
            url = get_url_from_gdrive_confirmation(res.text)
        except RuntimeError as e:
            logger.error("Access denied with the following error:")
            error = "\n".join(textwrap.wrap(str(e)))
            error = indent(error, "\t")
            logger.error("\n", error, "\n", file=sys.stderr)
            logger.error(
                "You may still be able to access the file from the browser:",
                file=sys.stderr,
            )
            logger.error("\n\t", url_origin, "\n", file=sys.stderr)
            return None, None

    if gdrive_file_id and is_gdrive_download_link:
        content_disposition = six.moves.urllib_parse.unquote(
            res.headers["Content-Disposition"]
        )
        m = re.search(r"filename\*=UTF-8''(.*)", content_disposition)
        filename_from_url = m.groups()[0]
        filename_from_url = filename_from_url.replace(os.path.sep, "_")
    else:
        filename_from_url = os.path.basename(url)

    return url, filename_from_url


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but indicates that we are
    # not using the MD5 checksum for cryptography. This enables its usage in restricted environments like FIPS. Without
    # it torchvision.datasets is unusable in these environments since we perform a MD5 check everywhere.
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _get_redirect_url(url: str, max_hops: int = 3) -> str:
    initial_url = url
    headers = {"Method": "HEAD", "User-Agent": USER_AGENT}

    for _ in range(max_hops + 1):
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers)
        ) as response:
            if response.url == url or response.url is None:
                return url

            url = response.url
    else:
        raise RecursionError(
            f"Request to {initial_url} exceeded {max_hops} redirects. The last redirect points to {url}."
        )


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def format_bytes(num_bytes):
    units = ["", "K", "M", "G", "T", "P"]
    unit_index = 0
    bytes_remaining = num_bytes
    while bytes_remaining >= 1024:
        bytes_remaining /= 1024.0
        unit_index += 1
    return f"{int(bytes_remaining * 10 + 0.5) / 10.0}{units[unit_index]}"