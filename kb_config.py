"""
kb_config.py - which MongoDB fields feed which knowledge-base files.

Written against the REAL schema printed by `python kb_build.py --inspect`
(collections: cities, hotels, rooms, rateplans). Nothing here is hotel data -
it is only a map of field names. kb_build.py writes only what it finds in
MongoDB, and only fields listed here (allow-list), so things like Email,
Mobile, whatsAppMobile, maxAgentDiscount, hotelogixId, propertyId, qrCode,
bookingEmails or any commission/agent field can never end up in a file the
voice agent might read aloud.

FIELD OPTIONS
  label         ("Text shown/spoken",)   English only
  paths         candidate Mongo paths (dotted); first one with a value wins
  kind          "text"    prose; a plain string or {"EN": "..."} object
                "neutral" times / numbers / address, used as-is
                "flag"    0/1/True -> "Available"; absent/0 -> says nothing
                          (we never claim "no gym": 0 may just mean unknown)
                "count"   integer > 0 -> "<n> <noun>"   (needs "noun")
                "lookup"  list of ids pointing into another collection
                          (needs "collection" and "name_paths")
  visible       path of an isVisible flag; if it is False the field is skipped
  details_only  True = kept in the "Details" section, never read aloud

Topics whose data does not exist in these four collections (check-in /
check-out times, child policy, payment methods, house rules, airport
distance ...) are intentionally not listed. If they live in another
collection (e.g. settings, cancellationpenalties, hotelairports), inspect it
with:  python kb_build.py --inspect --collections hotelairports
and add a field for it here.
"""

# Prices change constantly and Mongo answers them live (get_room_options), so
# they are kept OUT of the knowledge files. Leave False.
INCLUDE_PRICES = False


def _f(label, paths, kind="text", **extra):
    d = {"label": (label,), "paths": paths if isinstance(paths, list) else [paths],
         "kind": kind}
    d.update(extra)
    return d


# If a file's spoken text mentions one of these words, the tag is added so the
# okf_store topic matcher finds it.
AUTO_TAGS = {
    "breakfast": ["breakfast"],
    "wifi": ["wifi", "wi-fi"],
    "parking": ["parking"],
    "gym": ["gym", "fitness"],
    "pool": ["pool", "swimming"],
    "spa": ["spa"],
    "airport": ["airport"],
    "dining": ["restaurant", "dining", "food and beverage"],
    "child": ["child", "extra bed"],
}

# ----------------------------------------------------------------------------
# Hotel overview (hotels collection). The spoken answer is built from the hotel
# name, city, star rating, type and room types; these facts go to Details.
# ----------------------------------------------------------------------------
OVERVIEW_FIELDS = [
    _f("About", "description", details_only=True),
    _f("More about the hotel", "overview.EN.description",
       visible="overview.isVisible", details_only=True),
    _f("Star rating", "starRating", "neutral", details_only=True),
    _f("Hotel type", "hotelType", "neutral", details_only=True),
    _f("Total rooms in the hotel", "rooms", "neutral", details_only=True),
]

# ----------------------------------------------------------------------------
# Room types (rooms collection). Only rooms with a rate in the hotel's master
# rate plan are written - the same rule the booking flow uses.
# ----------------------------------------------------------------------------
ROOM_DESCRIPTION_FIELDS = [
    _f("Description", "description.EN"),
]
_NAME_PATHS = ["title.EN", "name.EN", "title", "name", "label.EN", "label"]

ROOM_DETAIL_FIELDS = [
    _f("Beds", "beds", "neutral"),
    _f("Bathrooms", "bathroom", "neutral"),
    _f("Maximum guests", "guest", "neutral"),
    _f("Room area (unit as stored in Mongo)", "area", "neutral"),
    _f("Rate name", "rateDetails.rateTitle.EN"),
    _f("Room amenities", "amenitiesIDs", "lookup",
       collection="amenities", name_paths=_NAME_PATHS),
]

# One "<room> policies" file per room type: cancellation and payment rules are
# stored per room in Mongo (rooms.rateDetails), so they are answered per room.
ROOM_POLICY_TAGS = ["policy", "cancellation", "cancel", "refund", "payment",
                    "guarantee", "advance"]
ROOM_POLICY_FIELDS = [
    _f("Cancellation policy", "rateDetails.cancellationPolicy.EN"),
    _f("Payment guarantee", "rateDetails.guaranteePolicy.EN"),
    _f("Rate inclusions", "rateDetails.rateDescription", details_only=True),
    _f("Member cancellation policy", "rateDetails.memberCancellationPolicy.EN",
       details_only=True),
    _f("Member payment guarantee", "rateDetails.memberGuaranteePolicy.EN",
       details_only=True),
    _f("Member benefits", "rateDetails.memberRateDescription", details_only=True),
]

# ----------------------------------------------------------------------------
# Hotel-level topics (hotels collection)
# ----------------------------------------------------------------------------
TOPICS = [
    {
        "path": "amenities/facilities.md",
        "type": "amenity",
        "title": "Hotel facilities and amenities",
        "tags": ["amenities", "facilities", "facility", "features", "services"],
        "spoken_style": "list",        # "Facilities include A, B and C."
        "fields": [
            _f("Swimming pool", "swimmingPool", "flag"),
            _f("Gym", "gym", "flag"),
            _f("Spa", "spa", "flag"),
            _f("Food and beverage outlets", "fbOutlet", "count",
               noun=("restaurant or food outlet", "restaurants or food outlets")),
            _f("Banquet halls", "banquetHalls", "count",
               noun=("banquet hall", "banquet halls")),
            # ids in hotels.amenitiesIDs / serviceIDs / wellnessIDs point into
            # these collections. If a name can't be read, kb_build.py says so;
            # run --inspect --collections amenities,services,wellnesses and
            # adjust name_paths.
            _f("Amenities", "amenitiesIDs", "lookup",
               collection="amenities", name_paths=_NAME_PATHS),
            _f("Services", "serviceIDs", "lookup",
               collection="services", name_paths=_NAME_PATHS),
            _f("Wellness", "wellnessIDs", "lookup",
               collection="wellnesses", name_paths=_NAME_PATHS),
        ],
    },
    {
        "path": "amenities/dining.md",
        "type": "amenity",
        "title": "Dining and restaurants",
        "tags": ["dining", "restaurant", "restaurants", "food", "meals"],
        "fields": [
            _f("Dining", "dining.EN.description", visible="dining.isVisible"),
        ],
    },
    {
        "path": "amenities/events-and-banquets.md",
        "type": "amenity",
        "title": "Events and banquets",
        "tags": ["events", "banquet", "venue", "conference", "wedding", "meeting"],
        "fields": [
            _f("Events and banquets", "venue.EN.description", visible="venue.isVisible"),
        ],
    },
    {
        "path": "amenities/wellness.md",
        "type": "amenity",
        "title": "Wellness and spa",
        "tags": ["wellness", "spa", "massage"],
        "fields": [
            _f("Wellness", "wellness.EN.description", visible="wellness.isVisible"),
        ],
    },
    {
        "path": "amenities/experiences.md",
        "type": "amenity",
        "title": "Activities and experiences",
        "tags": ["experiences", "activities", "things to do"],
        "fields": [
            _f("Experiences", "experiences.EN.description", visible="experiences.isVisible"),
        ],
    },
    {
        "path": "directions/how-to-reach.md",
        "type": "directions",
        "title": "Address and how to reach the hotel",
        "tags": ["directions", "address", "location", "how to reach", "map"],
        "fields": [
            _f("Address", "address", "neutral"),
            _f("Latitude", "latitude", "neutral", details_only=True),
            _f("Longitude", "longitude", "neutral", details_only=True),
            _f("Hotel website", "detailURL", "neutral", details_only=True),
        ],
    },
    {
        "path": "directions/nearby-attractions.md",
        "type": "directions",
        "title": "Nearby attractions",
        "tags": ["nearby", "attractions", "sightseeing", "places to visit"],
        "fields": [
            _f("Nearby attractions", "attraction.EN.description",
               visible="attraction.isVisible"),
        ],
    },
    {
        # The hotel's amenities, read from the `amenities` collection through
        # hotels.amenitiesIDs (names only - that collection has no descriptions).
        "path": "amenities/amenity-list.md",
        "type": "amenity",
        "title": "Hotel amenities (amenities collection)",
        "tags": ["amenities", "amenity list", "hotel amenities"],
        "extra": {"priority": "1"},   # on a tie with facilities.md, this one wins
        "auto_tags": False,        # no gym/pool/... auto tags: facilities.md answers those
        "spoken_style": "list",
        "fields": [
            _f("Amenities", "amenitiesIDs", "lookup",
               collection="amenities", name_paths=_NAME_PATHS),
        ],
    },
]

# ----------------------------------------------------------------------------
# Lists that live in their OWN collections - one file per hotel, one entry per
# row:   restaurants   -> amenities/restaurants.md
#        nearbyhotels  -> directions/nearby-places.md
#
# These two collections had not been inspected when this was written, so every
# field below is a CANDIDATE list (first path with a value wins). Run
#     python kb_build.py --inspect --collections restaurants,nearbyhotels,cuisines
# then correct any path. kb_build.py prints, per collection, how many rows it
# linked to a hotel and which fields it did NOT read - so a wrong guess shows
# up as a visible note instead of silently missing data.
# Only fields listed here are ever copied (allow-list); ids are never spoken.
# ----------------------------------------------------------------------------
HOTEL_LINK_PATHS = ["hotelId", "hotelID", "hotelIDs", "hotelIds", "hotel_id",
                    "hotel", "hotels"]

# A distance is only ever written WITH a unit. If your nearbyhotels distance is
# a bare number (e.g. 2.5) with no unit field, set the unit it is stored in
# (e.g. "km"); until then bare numbers are skipped - never guessed.
NEARBY_DISTANCE_UNIT = None

RELATED_LISTS = [
    {
        "collection": "restaurants",
        "path": "amenities/restaurants.md",
        "type": "amenity",
        "title": "Restaurants",
        "item_label": "Restaurant",
        "spoken_lead": "Restaurants include",
        "tags": ["restaurants", "restaurant", "dining", "food", "meals", "cuisine"],
        "hotel_paths": HOTEL_LINK_PATHS,
        "name": _f("Restaurant", ["name.EN", "title.EN", "restaurantName.EN",
                                  "name", "title", "restaurantName"]),
        "fields": [
            _f("Cuisine", ["cuisine.EN", "cuisineType.EN", "cuisine", "cuisineType"]),
            _f("Cuisine", ["cuisineIDs", "cuisinesIDs", "cuisines"], "lookup",
               collection="cuisines", name_paths=_NAME_PATHS),
            _f("Timings", ["timings.EN", "timing.EN", "openingHours.EN",
                           "timings", "timing", "openingHours"]),
            _f("Description", ["description.EN", "shortDescription.EN", "description"]),
            _f("Type", ["type", "restaurantType"]),
        ],
    },
    {
        "collection": "nearbyhotels",
        "path": "directions/nearby-places.md",
        "type": "directions",
        "title": "Nearby places",
        "item_label": "Place",
        "spoken_lead": "Nearby places include",
        "tags": ["nearby", "nearby places", "attractions", "sightseeing",
                 "places to visit", "distance"],
        "hotel_paths": HOTEL_LINK_PATHS,
        "name": _f("Place", ["name.EN", "title.EN", "hotelName.EN", "placeName.EN",
                             "name", "title", "hotelName", "placeName"]),
        # if each row only points at another hotel, take that hotel's name
        "name_fallbacks": [
            _f("Place", ["nearbyHotelID", "nearbyHotelId", "nearByHotelID"], "lookup",
               collection="hotels", name_paths=["name"]),
        ],
        # (field, unit that field is stored in or None); a bare number needs a unit
        "distance": [("distance", None), ("distanceKm", "km"), ("distanceInKm", "km"),
                     ("distance_km", "km"), ("distanceInKms", "km"), ("kms", "km"),
                     ("km", "km"), ("distanceMeters", "m"), ("distanceInMeters", "m")],
        "distance_unit_paths": ["unit", "distanceUnit", "distance_unit"],
        "fields": [
            _f("Description", ["description.EN", "shortDescription.EN", "description"]),
            _f("Address", ["address.EN", "address"], "neutral"),
            _f("Type", ["type", "category"]),
        ],
    },
]