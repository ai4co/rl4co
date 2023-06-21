import os
import re
import sys
import textwrap
import urllib
import warnings

from urllib.parse import parse_qs, urlparse

import requests

from rl4co.utils.download.constants import USER_AGENT
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def _parse_gdrive_url(url, warning=True):
    """Parse URLs especially for Google Drive links.
    file_id: ID of file on Google Drive.
    is_download_link: Flag if it is download link of Google Drive.
    """
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    is_gdrive = parsed.hostname in ["drive.google.com", "docs.google.com"]
    is_download_link = parsed.path.endswith("/uc")

    if not is_gdrive:
        return url, None, False

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


def get_url_filename_drive(url, sess, verify):
    url_origin = url
    url, gdrive_file_id, is_gdrive_download_link = _parse_gdrive_url(url)

    while True:
        try:
            res = sess.get(
                url, headers={"User-Agent": USER_AGENT}, stream=True, verify=verify
            )
        except requests.exceptions.ProxyError as e:
            log.error("An error has occurred using proxy:", sess.proxy, file=sys.stderr)
            log.error(e, file=sys.stderr)
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
            log.error("Access denied with the following error:")
            error = "\n".join(textwrap.wrap(str(e)))
            error = indent(error, "\t")
            log.error("\n", error, "\n", file=sys.stderr)
            log.error(
                "You may still be able to access the file from the browser:",
                file=sys.stderr,
            )
            log.error("\n\t", url_origin, "\n", file=sys.stderr)
            return None, None

    if gdrive_file_id and is_gdrive_download_link:
        content_disposition = urllib.parse.unquote(res.headers["Content-Disposition"])
        m = re.search(r"filename\*=UTF-8''(.*)", content_disposition)
        filename_from_url = m.groups()[0]
        filename_from_url = filename_from_url.replace(os.path.sep, "_")
    else:
        filename_from_url = os.path.basename(url)

    return url, filename_from_url


def indent(text, prefix):
    def prefixed_lines():
        for line in text.splitlines(True):
            yield (prefix + line if line.strip() else line)

    return "".join(prefixed_lines())
