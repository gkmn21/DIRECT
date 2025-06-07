
TIME_CONSTRAINTS = [2, 3, 4, 5, 6, 7, 8, 9, 10]

##
# OSM tags for tourist attractions, accommodations and food establishments
##
TOURIST_ATTR_TAGS = [
    ('tourism', 'artwork'),
    ('tourism', 'attraction'),
    ('tourism', 'viewpoint'),
    ('tourism', 'museum'),
    ('tourism', 'gallery'),
    ('tourism', 'picnic_site'),
    ('tourism', 'zoo'),
    ('tourism', 'theme_park'),
    ('tourism', 'aquarium'),
    ('tourism', 'wine_cellar'),
    ('tourism', 'casino'),
    ('tourism', 'highlight'),
    ('tourism', 'tower_viewer'),
    ('tourism', 'arts_centre'),
    ('tourism', 'history'),
    ('tourism', 'Atelier und Künstlerhaus'),
    ('tourism', 'clock'),
    ('tourism', 'zeitgenössische_Kunst'),
    ('tourism', 'shopping'),
    ('tourism', 'exhibition'),
    ('tourism', 'yes')
]

ACCOMMODATION_TAGS = [
    ('tourism', 'hotel'),
    ('tourism', 'hostel'),
    ('tourism', 'guest_house'),
    ('tourism', 'apartment'),
    ('tourism', 'camp_pitch'),
    ('tourism', 'camp_site'),
    ('tourism', 'caravan_site'),
    ('tourism', 'chalet'),
    ('tourism', 'motel'),
    ('tourism', 'spa_resort'),
    ('tourism', 'Serviced Apartments')
]

FOOD_ESTB_TAGS = [
    ('amenity', 'restaurant'),
    ('amenity', 'fast_food'),
    ('amenity', 'canteen'),
    ('amenity', 'food court'),
    ('fast food', 'cafeteria'),
    ('amenity', 'cafe')
]

NODE_CATEGORIES = [
    'junction', # category for non-poi nodes in network
    'accommodation',
    'food establishment', # where full meal/BLD can be eaten,
    'cafe',
    'artwork', # pois with ('tourism', 'artwork') tag or 'artwork_type' key
    'sculpture', # pois with ('artwork_type', 'sculpture'), ('attraction', 'sculpture'), ('artwork_type', 'sculpture_brick'), ('artwork_type', 'sculpture_group')
    'statue', # ('artwork_type', 'statue'), ('historic', 'statue')
    'graffiti and mural', # ('artwork_type', 'graffiti'), ('artwork_type', 'mural'), ('artwork_type', 'mural_painting')
    'museum', # with either key = museum or value = museum
    'gallery', # either key or value gallery, ('amenity', 'contemporary_art_gallery'), ('amenity', 'art_gallery'), ('gallery', 'photo')
    'historic landmark', # /'heritage'/'historic buildings' # either key or value 'historic', value = history, ('museum', 'art,history,nature')
    'natural wonder', # / 'natural sites' # key/value = natural
    'viewpoint', # key/value = viewpoint
    'other'
]

VISIT_DURATION_BASED_ON_CATEGORIES =  { # in seconds
    'junction': [0, 0],
    'accommodation': [0, 0],
    'food establishment': [1*3600, 2*3600], # where full meal/BLD can be eaten,
    'cafe': [1*3600, 2*3600],
    'artwork': [0.25 * 3600, 0.5 * 3600], # pois with ('tourism', 'artwork') tag or 'artwork_type' key
    'sculpture': [0.25 * 3600, 0.5 * 3600], # pois with ('artwork_type', 'sculpture'), ('attraction', 'sculpture'), ('artwork_type', 'sculpture_brick'), ('artwork_type', 'sculpture_group')
    'statue': [0.25 * 3600, 0.5 * 3600], # ('artwork_type', 'statue'), ('historic', 'statue')
    'graffiti and mural': [0.25*3600, 0.5 * 3600], # ('artwork_type', 'graffiti'), ('artwork_type', 'mural'), ('artwork_type', 'mural_painting')
    'museum': [1 * 3600, 3 * 3600], # with either key = museum or value = museum
    'gallery': [1 * 3600, 3 * 3600], # either key or value gallery, ('amenity', 'contemporary_art_gallery'), ('amenity', 'art_gallery'), ('gallery', 'photo')
    'historic landmark': [0.5 * 3600, 1 * 3600], # /'heritage'/'historic buildings' # either key or value 'historic', value = history, ('museum', 'art,history,nature')
    'natural wonder': [0.5 * 3600, 2 * 3600], # / 'natural sites' # key/value = natural
    'viewpoint': [0.5 * 3600, 1*3600], # key/value = viewpoint
    'other': [0.5*3600, 2*3600]
}