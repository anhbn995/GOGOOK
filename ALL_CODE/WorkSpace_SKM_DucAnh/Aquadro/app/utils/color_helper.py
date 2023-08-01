def hex_2_rgb(hex):
    hex = hex.replace('#', '')
    hlen = len(hex)
    return tuple(int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, int(hlen // 3)))


def rgb_2_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b)