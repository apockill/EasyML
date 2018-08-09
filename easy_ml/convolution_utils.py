def calc_deconv(filter_dim, stride_dim, input_dim, padding):
    """Calculate the output dimension after a deconvolution"""
    out_w = (input_dim[0] - 1) * stride_dim[0] + filter_dim[0] - 2 * padding - 1
    out_h = (input_dim[1] - 1) * stride_dim[1] + filter_dim[1] - 2 * padding - 1
    return out_w, out_h


def calc_conv(filter_dim, stride_dim, input_dim, padding):
    """Calculate the output dimension after a convolution"""
    out_w = (input_dim[0] - filter_dim[0] + 2 * padding) / stride_dim[0] + 1
    out_h = (input_dim[1] - filter_dim[1] + 2 * padding) / stride_dim[1] + 1
    return out_w, out_h
