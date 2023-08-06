
from model import KMeans
from utils import get_image, show_image, save_image, error
import os

def main():
    # get image
    image = get_image(r'D:\IISc\Sem2\ML\Assignments\A2\KMeans\original_image.jpg')
    img_shape = image.shape

    # reshape image
    image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    # create model
    num_clusters = 2 # modify k values
    kmeans = KMeans(num_clusters)

    # fit model
    kmeans.fit(image)

    # replace each pixel with its closest cluster center
    clustered_image = kmeans.replace_with_cluster_centers(image)  # Update the variable name

    # reshape image
    image_clustered = clustered_image.reshape(img_shape)

    # reshape original image
    original_image = image.reshape(img_shape)

    # Print the error
    print('MSE:', error(original_image, image_clustered))

    # show/save image
    # show_image(image)
    #save_image(image_clustered, f'image_clustered_{num_clusters}.jpg')
    output_path = os.path.join(r'D:\IISc\Sem2\ML\Assignments\A2\KMeans', f'image_clustered_{num_clusters}.jpg')
    save_image(image_clustered, output_path)

if __name__ == '__main__':
    main()
