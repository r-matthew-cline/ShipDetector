##############################################################################
##
## rbox_utils.py
##
## @author: Matthew Cline
## @version: 20180830
##
## Description: Set of utilities to find Intersection of Union and Angle
## Intersection of Union between bounding boxes.
##
##############################################################################

import cv2
import numpy as np

def union(rect_a, rect_b, shape=(768,768)):
	area_a = rect_a[1][0] * rect_a[1][1]
	area_b = rect_b[1][0] * rect_b[1][1]
	box_a = cv2.boxPoints(rect_a)
	box_b = cv2.boxPoints(rect_b)
	return area_a + area_b - intersection(box_a, box_b, shape)

def intersection(box_a, box_b, shape=(768,768)):
	img1 = np.zeros(shape)
	img2 = np.zeros(shape)
	cv2.drawContours(img1, [box_a], 0, color=1, thickness=-1)
	cv2.drawContours(img2, [box_b], 0, color=1, thickness=-1)
	mask = np.logical_and(img1, img2)
	_, contours, _ = cv2.findContours(mask.astype(uint8).copy(), 1, 2)
	cnt = contours[0]
	area = cv2.contourArea(cnt)
	return area


def IOU(rect1, rect2, shape=(768,768)):
	box1 = cv2.boxPoints(rect1)
	box2 = cv2.boxPoints(rect2)
	area_inter = intersection(box1, box2, shape)
	area_union = union(rect1, rect2, shape)
	return area_inter / area_union

def angle_IOU(rect1, rect2, shape=(768,768)):
	box1 = cv2.boxPoints(rect1)
	box2 = cv2.boxPoints(rect2)
	iou = IOU(rect1, rect2)
	ang1 = rect1[2]
	ang2 = rect2[2]
	return iou * abs(cos(ang1-ang2))
