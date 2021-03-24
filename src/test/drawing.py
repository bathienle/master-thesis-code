"""
Drawing tool for images
"""

import cv2
import numpy as np


class Painter:
    """
    Class to draw squiggles on an image.

    Attributes
    ----------
    filename : str
        The filename of the image.
    image : NumPy ndarray
        The image.
    signal : NumPy ndarray
        The squiggles drawn by the user.
    prev_x : int
        The previous x-coordinate on the image.
    prev_y : int
        The previous y-coordinate on the image.
    drawing : bool
        Whether to draw on the image or not

    Methods
    -------
    draw(event, x, y, *unused)
        Draw a line on an image.
    draw_on_image()
        Draw continuous points on an image.
    """

    def __init__(self, filename, image):
        self.filename = filename
        self.board = image.copy()  # Copy of the image to draw on

        # Create the empty signal map
        height, width, _ = image.shape
        self.signal = np.zeros((height, width), dtype=np.uint8)

        # Coordinates of the drawing
        self.prev_x = 0
        self.prev_y = 0

        # Whether the user is drawing or not
        self.drawing = False

        # List of list of squiggles coordinates
        self.squiggles = []

    def _draw(self, event, x, y, *unused):
        """
        Draw a line on the image.

        Parameters
        ----------
        event : int
            OpenCV event.
        x : int
            The x-coordinate of the mouse event.
        y : int
            The y-coordinate of the mouse event.
        unused : tuple
            Unused parameters.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.coordinates = [[x, y]]

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            prev_coord = (self.prev_x, self.prev_y)
            curr_coord = (x, y)
            cv2.line(self.board, prev_coord, curr_coord, (0, 0, 255), 1)
            cv2.line(self.signal, prev_coord, curr_coord, (255, 255, 255), 1)
            self.coordinates.append([x, y])

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.coordinates.append([x, y])
            self.squiggles.append(np.asarray(self.coordinates))

        self.prev_x = x
        self.prev_y = y

    def draw_on_image(self):
        """
        Draw continous lines on the image. 

        Return
        ------
        signal : NumPy ndarray
            The drawing performed by the user.
        squiggles : list
            The squiggles of the user.
        """
        cv2.namedWindow(self.filename)
        cv2.setMouseCallback(self.filename, self._draw)

        k = 0
        while(k != 27):
            cv2.imshow(self.filename, self.board)
            k = cv2.waitKey(1) & 0xFF

        cv2.destroyWindow(self.filename)

        return self.signal, self.squiggles
