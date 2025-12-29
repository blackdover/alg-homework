import math
from typing import Dict, Tuple


class GeoUtils:
    EARTH_RADIUS_M = 6371000.0
    KNOTS_TO_MPS = 0.514444

    @staticmethod
    def knots_to_mps(knots: float) -> float:
        return knots * GeoUtils.KNOTS_TO_MPS

    @staticmethod
    def deg_to_rad(degrees: float) -> float:
        return math.radians(degrees)

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat / 2) ** 2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return GeoUtils.EARTH_RADIUS_M * c

    @staticmethod
    def predict_position(lat_old: float, lon_old: float,
                        speed_mps: float, course_deg: float,
                        delta_t: float) -> Tuple[float, float]:
        distance = speed_mps * delta_t
        course_rad = math.radians(course_deg)
        lat_old_rad = math.radians(lat_old)
        lon_old_rad = math.radians(lon_old)
        dlat_rad = (distance * math.cos(course_rad)) / GeoUtils.EARTH_RADIUS_M
        dlon_rad = (distance * math.sin(course_rad)) / (GeoUtils.EARTH_RADIUS_M * math.cos(lat_old_rad))
        lat_new_rad = lat_old_rad + dlat_rad
        lon_new_rad = lon_old_rad + dlon_rad
        lat_new = math.degrees(lat_new_rad)
        lon_new = math.degrees(lon_new_rad)
        return lat_new, lon_new

    @staticmethod
    def point_to_line_distance(lat: float, lon: float,
                              lat1: float, lon1: float,
                              lat2: float, lon2: float) -> float:
        avg_lat = (lat1 + lat2) / 2
        cos_lat = math.cos(math.radians(avg_lat))
        x = lon * 111000 * cos_lat
        y = lat * 111000
        x1 = lon1 * 111000 * cos_lat
        y1 = lat1 * 111000
        x2 = lon2 * 111000 * cos_lat
        y2 = lat2 * 111000
        if x1 == x2 and y1 == y2:
            return math.sqrt((x - x1)**2 + (y - y1)**2)
        dx = x2 - x1
        dy = y2 - y1
        px = x - x1
        py = y - y1
        t = max(0, min(1, (px * dx + py * dy) / (dx * dx + dy * dy)))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        return math.sqrt((x - closest_x)**2 + (y - closest_y)**2)


