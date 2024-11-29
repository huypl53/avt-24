import math
from datetime import datetime, timedelta
from typing import Any, Iterable, List

from sqlalchemy import select, text

from app.db.connector import get_db
from app.model.task import AdsbMd, TaskMd

sigma_cog = 3
sigma_width = 5
sigma_height = 5
sigma_position = 10
likelihood_position_threshold = 100


def likelihood_cog(cog_detected, cog_ais, time_sec):
    term1 = math.exp(
        -0.5 * ((cog_detected - cog_ais) / (sigma_cog + 0.2 * time_sec)) ** 2
    )
    term2 = math.exp(
        -0.5 * ((cog_detected + 180 - cog_ais) / (sigma_cog + 0.2 * time_sec)) ** 2
    )
    return term1 + term2


def likelihood_width(width_detected, width_ais):
    return math.exp(-0.5 * ((width_detected - width_ais) / sigma_width) ** 2)


def likelihood_height(height_detected, height_ais):
    return math.exp(-0.5 * ((height_detected - height_ais) / sigma_height) ** 2)


def likelihood_position(
    position_detected_lat,
    position_detected_lon,
    position_ais_lat,
    position_ais_lon,
    time_sec,
):
    distance = math.sqrt(
        (position_ais_lat - position_detected_lat) ** 2
        + (position_ais_lon - position_detected_lon) ** 2
    )
    return math.exp(-0.5 * (distance / (sigma_position + time_sec * 10)) ** 2)


def total_likelihood(likelihood_cog_value, likelihood_width_value, *other_likelihoods):
    result = likelihood_cog_value * likelihood_width_value
    for likelihood in other_likelihoods:
        result *= likelihood
    return result


class Matching_Image:
    def __init__(self):
        pass


def position_check(
    position_detected_lat,
    position_detected_lon,
    position_ais_lat,
    position_ais_lon,
    time_sec,
):
    if (
        likelihood_position(
            position_detected_lat,
            position_detected_lon,
            position_ais_lat,
            position_ais_lon,
            time_sec,
        )
        > likelihood_position_threshold
    ):
        return True
    else:
        return False


def likelihood(
    cog_detected,
    cog_ais,
    width_detected,
    width_ais,
    height_detected,
    height_ais,
    position_detected_lat,
    position_detected_lon,
    position_ais_lat,
    position_ais_lon,
    time_sec,
):
    likelihood_cog_value = likelihood_cog(cog_detected, cog_ais, time_sec)
    likelihood_width_value = likelihood_width(width_detected, width_ais)
    likelihood_height_value = likelihood_height(height_detected, height_ais)
    likelihood_position_value = likelihood_position(
        position_detected_lat,
        position_detected_lon,
        position_ais_lat,
        position_ais_lon,
        time_sec,
    )
    total_likelihood_value = total_likelihood(
        likelihood_cog_value,
        likelihood_width_value,
        likelihood_position_value,
        likelihood_height_value,
    )
    return total_likelihood_value


async def check_adsb(
    ships_coords: Iterable[Any],
    session=None,
) -> List[int] | None:
    try:
        time_detected = datetime.now()
        if not session:
            a_session = anext(get_db("adsb_object"))
            session = await a_session
        query = text(
            f"""
            SELECT *
            FROM avt_adsb_data
            WHERE import_at BETWEEN {time_detected - timedelta(seconds=10)} AND {time_detected + timedelta(seconds=10)}
        """
        )
        results = await session.execute(query)
        # result = adsbs.first()
        mapping_results = results.mappings().all()
        adsbs: List[AdsbMd] = [m["AdsbMd"] for m in mapping_results]
        if adsbs is None:
            return

        valid_idx: List[int] = []
        for coords in ships_coords:
            (
                latitude_detected,
                longitude_detected,
                width_detected,
                height_detected,
                cog_detected,
            ) = coords[:5]
            for i in range(len(adsbs)):
                adsb_record: AdsbMd = adsbs[i]
                ship_id = adsb_record["id"]
                longitude_ais = adsb_record["lng"]
                latitude_ais = adsb_record["lat"]
                width_ais = adsb_record["width"]
                height_ais = adsb_record["height"]
                cog_ais = adsb_record["cog"]
                if not position_check(
                    latitude_detected,
                    longitude_detected,
                    longitude_ais,
                    latitude_ais,
                    time_detected,
                ):
                    continue
                valid_idx.append(i)
                # list_possible_ship.append([ship_id, likelihood(cog_detected, cog_ais, width_detected,
                #                                                                 width_ais, height_detected,
                #                                                                 height_ais, latitude_detected,
                #                                                                 longitude_detected, latitude_ais,
                #                                                                 longitude_ais, time_detected)])
        # task: TaskMd = result
        return valid_idx
    except:
        pass
