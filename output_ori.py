from utils import *

test_coordinates, test_ncoordinates = load_single_cat_h5("computer", 4096, "test", "coordinates", "ncoordinates")
test_masks = np.random.choice(test_coordinates.shape[0], 5, replace=False)

batch_test_coordinates = test_coordinates[test_masks]
batch_test_ncoordinates = test_ncoordinates[test_masks]

for idx, data in enumerate(batch_test_coordinates):
    for nidx, ndata in enumerate(batch_test_ncoordinates):
        if idx == nidx:
            print(data.shape)
            display(ndata, data, name_='original/{}.png'.format(idx+1))
            print("Success save the {} picture".format(idx+1))
