"""Project TABLE-plane geometry into the calibrated image for OpenCV drawing."""
import math


def project_vector(world, x_cm, y_cm, vx_cm_s, vy_cm_s, seconds=1.0):
    return (world.to_pixel(x_cm, y_cm),
            world.to_pixel(x_cm + vx_cm_s * seconds,
                           y_cm + vy_cm_s * seconds))


def project_circle(world, x_cm, y_cm, radius_cm, samples=32):
    if not math.isfinite(radius_cm) or radius_cm < 0 or samples < 8:
        raise ValueError("invalid TABLE circle")
    return [world.to_pixel(x_cm + radius_cm * math.cos(2 * math.pi * n / samples),
                           y_cm + radius_cm * math.sin(2 * math.pi * n / samples))
            for n in range(samples)]


def cv_points(points):
    return [tuple(int(round(value)) for value in point) for point in points]
