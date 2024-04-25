def default_2Dcoords(func):
    def wrapper(I2D_obj, theta = None, phi = None, *args, **kwargs):
        
        theta = 90 - I2D_obj.lat if theta is None else theta
        phi   = I2D_obj.lon      if phi   is None else phi

        return func(I2D_obj, theta = theta, phi = phi, deg = True, *args, **kwargs)

    return wrapper

def default_3Dcoords(func):
    def wrapper(I2D_obj, r = None, theta = None, phi = None, *args, **kwargs):
        
        r     = I2D_obj.RI       if r     is None else r
        theta = 90 - I2D_obj.lat if theta is None else theta
        phi   = I2D_obj.lon      if phi   is None else phi

        return func(I2D_obj, r = r, theta = theta, phi = phi, deg = True, *args, **kwargs)
        
    return wrapper
