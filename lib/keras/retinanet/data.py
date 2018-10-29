import numpy as np
import lib.keras.retinanet.utils as k_utils
import cv2
import threading
import warnings
import random
import keras
import os
import PIL
import six
import xml.etree.cElementTree as ET

class TransformParameters:
    """ Struct holding parameters determining how to apply a transformation to an image.

    Args
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """
    def __init__(
            self,
            fill_mode            = 'nearest',
            interpolation        = 'linear',
            cval                 = 0,
            relative_translation = True,
    ):
        self.fill_mode            = fill_mode
        self.cval                 = cval
        self.interpolation        = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4

class Generator(object):
    """ Abstract generator class.
    """

    def __init__(
            self,
            transform_generator = k_utils.random_transform_generator(flip_x_chance=0.5),
            batch_size=1,
            group_method='ratio',  # one of 'none', 'random', 'ratio'
            shuffle_groups=True,
            image_min_side=800,
            image_max_side=1333,
            transform_parameters=None,
            compute_anchor_targets=k_utils.anchor_targets_bbox,
            compute_shapes=k_utils.guess_shapes,
            preprocess_image=k_utils.preprocess_image,
            config=None
    ):
        """ Initialize Generator object.

        Args
            transform_generator    : A generator used to randomly transform images and annotations.
            batch_size             : The size of the batches to generate.
            group_method           : Determines how images are grouped together (defaults to 'ratio', one of ('none', 'random', 'ratio')).
            shuffle_groups         : If True, shuffles the groups each epoch.
            image_min_side         : After resizing the minimum side of an image is equal to image_min_side.
            image_max_side         : If after resizing the maximum side is larger than image_max_side, scales down further so that the max side is equal to image_max_side.
            transform_parameters   : The transform parameters used for data augmentation.
            compute_anchor_targets : Function handler for computing the targets of anchors for an image and its annotations.
            compute_shapes         : Function handler for computing the shapes of the pyramid for a given input.
            preprocess_image       : Function handler for preprocessing an image (scaling / normalizing) for passing through a network.
        """
        self.transform_generator    = transform_generator
        self.batch_size             = int(batch_size)
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.transform_parameters   = transform_parameters or TransformParameters()
        self.compute_anchor_targets = compute_anchor_targets
        self.compute_shapes         = compute_shapes
        self.preprocess_image       = preprocess_image
        self.config                 = config

        self.group_index = 0
        self.lock        = threading.Lock()

        self.group_images()

    def size(self):
        """ Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """ Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """ Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """ Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        """ Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        for annotations in annotations_group:
            assert(isinstance(annotations, dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(type(annotations))
            assert('labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert('bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def filter_annotations(self, image_group, annotations_group, group):
        """ Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        """ Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            if transform is None:
                transform = k_utils.adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

            # apply transformation to image
            image = k_utils.apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations['bboxes'] = annotations['bboxes'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = k_utils.transform_aabb(transform, annotations['bboxes'][index, :])

        return image, annotations

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return k_utils.resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        # preprocess the image
        image = self.preprocess_image(image)

        # randomly transform image and annotations
        image, annotations = self.random_transform_group_entry(image, annotations)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= image_scale

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # preprocess a single group entry
            image, annotations = self.preprocess_group_entry(image, annotations)

            # copy processed data back to group
            image_group[index]       = image
            annotations_group[index] = annotations

        return image_group, annotations_group

    def group_images(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        if keras.backend.image_data_format() == 'channels_first':
            image_batch = image_batch.transpose((0, 3, 1, 2))

        return image_batch

    def generate_anchors(self, image_shape):
        anchor_params = None
        if self.config and 'anchor_parameters' in self.config:
            anchor_params = k_utils.parse_anchor_parameters(self.config)
        return k_utils.anchors_for_shape(image_shape, anchor_params=anchor_params, shapes_callback=self.compute_shapes)

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        anchors   = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes()
        )

        return list(batches)

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)


class PascalVocGenerator(Generator):
    """ Generate data for a Pascal VOC dataset.

    See http://host.robots.ox.ac.uk/pascal/VOC/ for more information.
    """

    def __init__(
            self,
            data_dir,
            set_name,
            classes={
                'aeroplane'   : 0,
                'bicycle'     : 1,
                'bird'        : 2,
                'boat'        : 3,
                'bottle'      : 4,
                'bus'         : 5,
                'car'         : 6,
                'cat'         : 7,
                'chair'       : 8,
                'cow'         : 9,
                'diningtable' : 10,
                'dog'         : 11,
                'horse'       : 12,
                'motorbike'   : 13,
                'person'      : 14,
                'pottedplant' : 15,
                'sheep'       : 16,
                'sofa'        : 17,
                'train'       : 18,
                'tvmonitor'   : 19
            },
            image_extension='.jpg',
            skip_truncated=False,
            skip_difficult=False,
            **kwargs
    ):
        """ Initialize a Pascal VOC data generator.

        Args
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
            csv_class_file: Path to the CSV classes file.
        """
        self.data_dir             = data_dir
        self.set_name             = set_name
        self.classes              = classes
        self.image_names          = [l.strip().split(None, 1)[0] for l in open(os.path.join(data_dir, 'MainSet', set_name + '.txt')).readlines()]
        self.image_extension      = image_extension
        self.skip_truncated       = skip_truncated
        self.skip_difficult       = skip_difficult

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(PascalVocGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return len(self.classes)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        path  = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        image = PIL.Image.open(path)
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        return k_utils.read_image_bgr(path)

    def __parse_annotation(self, element):
        """ Parse an annotation given an XML element.
        """
        truncated = k_utils._findNode(element, 'truncated', parse=int)
        difficult = k_utils._findNode(element, 'difficult', parse=int)

        class_name = k_utils._findNode(element, 'name').text
        if class_name not in self.classes:
            raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(self.classes.keys())))

        box = np.zeros((4,))
        label = self.name_to_label(class_name)

        bndbox    = k_utils._findNode(element, 'bndbox')
        box[0] = k_utils._findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
        box[1] = k_utils._findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
        box[2] = k_utils._findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
        box[3] = k_utils._findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1

        return truncated, difficult, box, label

    def __parse_annotations(self, xml_root):
        """ Parse all annotations under the xml_root.
        """
        annotations = {'labels': np.empty((len(xml_root.findall('object')),)), 'bboxes': np.empty((len(xml_root.findall('object')), 4))}
        for i, element in enumerate(xml_root.iter('object')):
            try:
                truncated, difficult, box, label = self.__parse_annotation(element)
            except ValueError as e:
                six.raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

            if truncated and self.skip_truncated:
                continue
            if difficult and self.skip_difficult:
                continue

            annotations['bboxes'][i, :] = box
            annotations['labels'][i] = label

        return annotations

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        filename = self.image_names[image_index] + '.xml'
        try:
            tree = ET.parse(os.path.join(self.data_dir, 'Annotations', filename))
            return self.__parse_annotations(tree.getroot())
        except ET.ParseError as e:
            six.raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            six.raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)