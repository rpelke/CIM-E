import re


def parse_model_string(file_name: str) -> dict:
    """Parse the model file name to extract the model data.
    Args:
        file_name (str): file name (without path and extension)
    Returns:
        dict: Parameter dict
    """
    pattern = r"(?P<type>^[a-z]+)_(?P<data>[a-z0-9]+)_(?P<model>[a-zA-Z0-9]+)_(?P<batch>b\d+)_mxn(?P<xbar>\d+x\d+)_inp(?P<ifm>\d+x\d+x\d+x\d+)"
    match = re.match(pattern, file_name)
    if match:
        raw_data = match.groupdict()

        ifm_shape_str = raw_data['ifm']
        ifm_shape_array = list(map(int, ifm_shape_str.split('x')))
        raw_data['ifm'] = ifm_shape_array

        xbar_dim_str = raw_data['xbar']
        xbar_dim_array = list(map(int, xbar_dim_str.split('x')))
        raw_data['xbar'] = xbar_dim_array

        return raw_data
    else:
        raise ValueError("Unknown format.")
