const useState = React.useState;

function getComponentProps(node) {
  const attributeNames = node.getAttributeNames();
  const props = {};
  for (const attributeName of attributeNames) {
    if (attributeName.startsWith("data-")) {
      const propName = attributeName.slice(5);
      props[propName] = node.getAttribute(attributeName);
    }
  }
  return props;
}

function mountComponent(querySelector, Component) {
  const containers = document.querySelectorAll(querySelector);
  for (const container of containers) {
    const props = getComponentProps(container);
    const root = ReactDOM.createRoot(container);
    root.render(<Component {...props} />);
  }
}

// LikeButtonWithTitle Component

function LikeButtonWithTitle({ title, margin, padding }) {
  const [likeCount, setLikeCount] = useState(100500);
  return (
    <button onClick={() => setLikeCount(likeCount + 1)} style={{ margin, padding }}>
      Like {title} {likeCount}
    </button>
  );
}

mountComponent(".LikeButtonWithTitle", LikeButtonWithTitle);


// ReactGreeter component

function ReactGreeter() {
  const [name, setName] = useState("");
  const onSubmit = (event) => {
    event.preventDefault();
    alert(`Hello, ${name}!`);
  };
  return (
    <form onSubmit={onSubmit}>
      <input
        type="text"
        placeholder="Enter your name"
        required
        value={name}
        onChange={(event) => setName(event.target.value)}
      />
      <button type="submit" disabled={!name}>Submit</button>
    </form>
  );
}

mountComponent(".ReactGreeter", ReactGreeter);
