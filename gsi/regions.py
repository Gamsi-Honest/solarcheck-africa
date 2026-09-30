"""
Regional intelligence for GSI mode.

SolarCheck started in Cameroon; GSI mode makes the reasoning honest about
*where* the panel is. Two identical panels are a good deal in Yaoundé and a
bad one in Garoua, because the solar resource, the grid tariff and the
counterfeit supply chain all differ.

Data provenance:
  * Specific yield (kWh/kWp/year) and peak-sun-hours: PVGIS-class long-term
    averages for the site, de-rated ~14% for soiling, cabling, inverter and
    temperature losses. Values are engineering estimates, not measurements —
    they are labelled as such in the UI.
  * Tariffs: published ENEO/ARSEL residential brackets for Cameroon
    (Nov-2024 harmonised structure: 50 / 79 / 94 / 99 XAF per kWh across the
    0-110 / 111-220 / 221-400 / 400+ kWh bands) and published national
    averages elsewhere. `tariff_per_kwh` is the blended residential rate.
  * `price_band` is the honest retail range in local currency per watt-peak
    for a genuine new module, triangulated from importer/distributor listings
    in the region. It is a sanity band, not a quote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Region:
    key: str
    name: str
    country: str
    currency: str
    currency_symbol: str
    peak_sun_hours: float  # usable equivalent hours per day, system-level
    specific_yield: float  # kWh per kWp per year, system-level
    tariff_per_kwh: float  # blended residential tariff, local currency
    tariff_tiers: List[tuple] = field(default_factory=list)  # (kWh_upper, price)
    grid_reliability: float = 0.7  # 0..1, how dependable the grid is
    price_band: Optional[tuple] = None  # (low, high) local currency per Wp
    counterfeit_prevalence: float = 0.35  # 0..1 market prior of fakes in circulation
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "country": self.country,
            "currency": self.currency,
            "currency_symbol": self.currency_symbol,
            "peak_sun_hours": self.peak_sun_hours,
            "specific_yield": self.specific_yield,
            "tariff_per_kwh": self.tariff_per_kwh,
            "grid_reliability": self.grid_reliability,
            "price_band": self.price_band,
            "counterfeit_prevalence": self.counterfeit_prevalence,
            "notes": self.notes,
        }


REGIONS: List[Region] = [
    Region(
        "yaounde", "Yaoundé", "Cameroon", "XAF", "FCFA",
        peak_sun_hours=3.9, specific_yield=1430, tariff_per_kwh=85.0,
        tariff_tiers=[(110, 50), (220, 79), (400, 94), (10**9, 99)],
        grid_reliability=0.72, price_band=(230, 420), counterfeit_prevalence=0.38,
        notes="Reference market. Two rainy seasons and frequent afternoon convection "
              "make the usable solar window shorter than the map suggests.",
    ),
    Region(
        "douala", "Douala", "Cameroon", "XAF", "FCFA",
        peak_sun_hours=3.4, specific_yield=1250, tariff_per_kwh=85.0,
        tariff_tiers=[(110, 50), (220, 79), (400, 94), (10**9, 99)],
        grid_reliability=0.62, price_band=(230, 420), counterfeit_prevalence=0.42,
        notes="Coastal humidity and haze; expect salt-mist soiling on rooftop arrays.",
    ),
    Region(
        "bafoussam", "Bafoussam", "Cameroon", "XAF", "FCFA",
        peak_sun_hours=3.9, specific_yield=1420, tariff_per_kwh=85.0,
        grid_reliability=0.58, price_band=(230, 430), counterfeit_prevalence=0.40,
        notes="Highland market; long-distance transport adds to landed panel cost.",
    ),
    Region(
        "bamenda", "Bamenda", "Cameroon", "XAF", "FCFA",
        peak_sun_hours=4.2, specific_yield=1510, tariff_per_kwh=85.0,
        grid_reliability=0.52, price_band=(240, 440), counterfeit_prevalence=0.44,
        notes="Extended grid outages make solar an availability play, not just a cost play.",
    ),
    Region(
        "garoua", "Garoua", "Cameroon", "XAF", "FCFA",
        peak_sun_hours=5.5, specific_yield=1690, tariff_per_kwh=85.0,
        grid_reliability=0.55, price_band=(220, 400), counterfeit_prevalence=0.34,
        notes="Sahelian belt — highest Cameroon yield; dust soiling is the main loss term.",
    ),
    Region(
        "maroua", "Maroua", "Cameroon", "XAF", "FCFA",
        peak_sun_hours=5.7, specific_yield=1730, tariff_per_kwh=85.0,
        grid_reliability=0.48, price_band=(220, 400), counterfeit_prevalence=0.36,
        notes="Extreme Far North yield. Clean panels every 2 weeks in dry season.",
    ),
    Region(
        "kribi", "Kribi", "Cameroon", "XAF", "FCFA",
        peak_sun_hours=3.6, specific_yield=1320, tariff_per_kwh=85.0,
        grid_reliability=0.60, price_band=(240, 430), counterfeit_prevalence=0.38,
    ),
    Region(
        "ngaoundere", "Ngaoundéré", "Cameroon", "XAF", "FCFA",
        peak_sun_hours=5.1, specific_yield=1600, tariff_per_kwh=85.0,
        grid_reliability=0.50, price_band=(230, 420), counterfeit_prevalence=0.36,
    ),
    Region(
        "dakar", "Dakar", "Senegal", "XOF", "FCFA",
        peak_sun_hours=5.3, specific_yield=1650, tariff_per_kwh=115.0,
        grid_reliability=0.78, price_band=(210, 380), counterfeit_prevalence=0.28,
        notes="Mature import market; higher tariff makes payback short.",
    ),
    Region(
        "abidjan", "Abidjan", "Côte d'Ivoire", "XOF", "FCFA",
        peak_sun_hours=4.4, specific_yield=1370, tariff_per_kwh=90.0,
        grid_reliability=0.74, price_band=(215, 390), counterfeit_prevalence=0.31,
    ),
    Region(
        "lagos", "Lagos", "Nigeria", "NGN", "₦",
        peak_sun_hours=4.5, specific_yield=1390, tariff_per_kwh=225.0,
        grid_reliability=0.55, price_band=None, counterfeit_prevalence=0.45,
        notes="Band A tariff regime plus deep diesel dependence; verify price locally.",
    ),
    Region(
        "accra", "Accra", "Ghana", "GHS", "GH₵",
        peak_sun_hours=4.7, specific_yield=1450, tariff_per_kwh=1.9,
        grid_reliability=0.68, price_band=None, counterfeit_prevalence=0.33,
    ),
    Region(
        "nairobi", "Nairobi", "Kenya", "KES", "KSh",
        peak_sun_hours=5.0, specific_yield=1520, tariff_per_kwh=27.0,
        grid_reliability=0.80, price_band=None, counterfeit_prevalence=0.25,
    ),
    Region(
        "kigali", "Kigali", "Rwanda", "RWF", "FRw",
        peak_sun_hours=4.8, specific_yield=1440, tariff_per_kwh=250.0,
        grid_reliability=0.70, price_band=None, counterfeit_prevalence=0.24,
    ),
    Region(
        "kinshasa", "Kinshasa", "DR Congo", "CDF", "FC",
        peak_sun_hours=4.3, specific_yield=1330, tariff_per_kwh=180.0,
        grid_reliability=0.45, price_band=None, counterfeit_prevalence=0.48,
        notes="Very high counterfeit prevalence reported by installers.",
    ),
    Region(
        "dar_es_salaam", "Dar es Salaam", "Tanzania", "TZS", "TSh",
        peak_sun_hours=5.0, specific_yield=1490, tariff_per_kwh=320.0,
        grid_reliability=0.62, price_band=None, counterfeit_prevalence=0.33,
    ),
    Region(
        "kampala", "Kampala", "Uganda", "UGX", "USh",
        peak_sun_hours=4.8, specific_yield=1430, tariff_per_kwh=780.0,
        grid_reliability=0.58, price_band=None, counterfeit_prevalence=0.34,
    ),
    Region(
        "lusaka", "Lusaka", "Zambia", "ZMW", "ZK",
        peak_sun_hours=5.4, specific_yield=1620, tariff_per_kwh=2.2,
        grid_reliability=0.60, price_band=None, counterfeit_prevalence=0.30,
    ),
    Region(
        "addis", "Addis Ababa", "Ethiopia", "ETB", "Br",
        peak_sun_hours=5.5, specific_yield=1640, tariff_per_kwh=4.5,
        grid_reliability=0.55, price_band=None, counterfeit_prevalence=0.32,
    ),
    Region(
        "niamey", "Niamey", "Niger", "XOF", "FCFA",
        peak_sun_hours=5.9, specific_yield=1790, tariff_per_kwh=110.0,
        grid_reliability=0.45, price_band=(215, 400), counterfeit_prevalence=0.37,
        notes="One of the strongest solar resources on the continent.",
    ),
]

REGION_MAP: Dict[str, Region] = {r.key: r for r in REGIONS}
DEFAULT_REGION = "yaounde"


def get_region(key: Optional[str]) -> Region:
    if key and key in REGION_MAP:
        return REGION_MAP[key]
    return REGION_MAP[DEFAULT_REGION]


def labels() -> Dict[str, str]:
    return {r.key: f"{r.name}, {r.country}" for r in REGIONS}
