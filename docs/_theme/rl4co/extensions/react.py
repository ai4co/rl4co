def get_react_component_rst(class_name, **props):
    data_props = " ".join([f'data-{key}="{value}"' for key, value in props.items()])
    return f"""
.. raw:: html

    <div class="{class_name}" {data_props}></div>
    """
