

FINAL_CATEGORIES = [
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