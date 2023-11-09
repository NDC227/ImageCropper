# ImageCropper
This is a little project that I was inspired to create when designing a tier list (of Taiwanese street food) for fun. Tier lists in modern pop culture are designed as grid-like charts, with each row corresponding to a given "tier", going from A or S at the top to D or F at the bottom. The objects in this grid are typically not textual but rather images, which are set to fit in the grid uniformly. This formatting of images in a grid proved to be a problem to me when I wanted to quickly throw together images in a tier list in PowerPoint (rather than using an online tier list maker) - images come in various different aspect ratios and don't align without cropping.

I set out to create a tool that would automatically crop an image down to a square around the focal matter of the image (focal in terms of important object, not necessarily camera focus). Using a pipeline that incorporates various computer vision techniques with OpenCV, I created this prototype that could easily be imported in a full-blown implementation of a tier list maker. The pipeline is designed as such:

## Pipeline
1. Take in an image and preprocess it by setting it to grayscale and applying a Gaussian blur to denoise.
2. Extract edges using the Canny algorithm and dilate the edges, to further denoise. These edges should more or less line up to objects.
3. Obtain contour objects using the extracted edges to perform geometry calculations.
4. Filter out contours based on their area.
5. Find the centroids of each of the contour objects, and use Agglomerative Clustering to group the centroids (and simultaneously, the contours).
6. Pick the "best" cluster of contours by choosing the cluster with the largest aggregate contour area.
7. Find the centroid of the "best" cluster and crop the image around this point, adjusting to the image bounds.

## Example
A sample execution of this script on the following input produces the desired result:

Input:

![oyster_omelet](https://github.com/NDC227/ImageCropper/assets/72755235/59c299de-77b1-46da-b570-af8be803ab60)

Output:

![oyster_omelet](https://github.com/NDC227/ImageCropper/assets/72755235/7445a3e8-8de8-4255-89c4-d0005820c645)
