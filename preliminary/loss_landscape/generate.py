# this is based on https://github.com/marcellodebernardi/loss-landscapes


import copy
import typing
import torch.nn
import numpy as np
from .model_wrapper import ModelWrapper, wrap_model
from .model_parameters import rand_u_like, orthogonal_to
from loss_landscapes.metrics.metric import Metric


def three_models(model1, model2, model3, model4, metric, distance=1, steps=20,
                 normalization='filter', deepcopy_model=False):
    model_start_wrapper1 = wrap_model(copy.deepcopy(model1) if deepcopy_model else model1)
    model_start_wrapper2 = wrap_model(copy.deepcopy(model2) if deepcopy_model else model2)
    model_start_wrapper3 = wrap_model(copy.deepcopy(model3) if deepcopy_model else model3)
    model_start_wrapper4 = wrap_model(copy.deepcopy(model4) if deepcopy_model else model4)

    parameters1 = model_start_wrapper1.get_module_parameters()
    parameters2 = model_start_wrapper2.get_module_parameters()
    parameters3 = model_start_wrapper3.get_module_parameters()
    parameters4 = model_start_wrapper4.get_module_parameters()

    start_point = (parameters1 + parameters2 + 4 * parameters3) / 6
    # print(start_point)

    dir_one, dir_two = get_directions(normalization, start_point)

    vector1 = parameters1 - start_point
    vector2 = parameters2 - start_point
    vector3 = parameters3 - start_point
    vector4 = parameters4 - start_point

    coor1 = project_vec_into(vector1, dir_one, dir_two)
    coor2 = project_vec_into(vector2, dir_one, dir_two)
    coor3 = project_vec_into(vector3, dir_one, dir_two)
    coor4 = project_vec_into(vector4, dir_one, dir_two)
    print(coor1)
    print(coor2)
    print(coor3)
    print(coor4)

    # scale to match steps and total distance
    dir_one.mul_(((start_point.model_norm() / distance) / steps) / dir_one.model_norm())
    dir_two.mul_(((start_point.model_norm() / distance) / steps) / dir_two.model_norm())
    exit()

    # Move start point so that original start params will be in the center of the plot
    dir_one.mul_(steps / 2)
    dir_two.mul_(steps / 2)
    start_point.sub_(dir_one)
    start_point.sub_(dir_two)
    dir_one.truediv_(steps / 2)
    dir_two.truediv_(steps / 2)


def project_vec_into(vec, dir_one, dir_two):
    coor_one = dir_one.project(vec)
    coor_two = dir_two.project(vec)
    return coor_one, coor_two


def random_plane(model: typing.Union[torch.nn.Module, ModelWrapper], metric: Metric, distance=1, steps=20,
                 normalization='filter', deepcopy_model=False) -> np.ndarray:
    """
    Returns the computed value of the evaluation function applied to the model or agent along a planar
    subspace of the parameter space defined by a start point and two randomly sampled directions.
    The models supplied can be either torch.nn.Module models, or ModelWrapper objects
    from the loss_landscapes library for more complex cases.

    That is, given a neural network model, whose parameters define a point in parameter
    space, and a distance, the loss is computed at 'steps' * 'steps' points along the
    plane defined by the two random directions, from the start point up to the maximum
    distance in both directions.

    Note that the dimensionality of the model parameters has an impact on the expected
    length of a uniformly sampled other in parameter space. That is, the more parameters
    a model has, the longer the distance in the random other's direction should be,
    in order to see meaningful change in individual parameters. Normalizing the
    direction other according to the model's current parameter values, which is supported
    through the 'normalization' parameter, helps reduce the impact of the distance
    parameter. In future releases, the distance parameter will refer to the maximum change
    in an individual parameter, rather than the length of the random direction other.

    Note also that a simple planar approximation with randomly sampled directions can produce
    misleading approximations of the loss landscape due to the scale invariance of neural
    networks. The sharpness/flatness of minima or maxima is affected by the scale of the neural
    network weights. For more details, see `https://arxiv.org/abs/1712.09913v3`. It is
    recommended to normalize the directions, preferably with the 'filter' option.

    The Metric supplied has to be a subclass of the loss_landscapes.metrics.Metric class,
    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    :param model: the model defining the origin point of the plane in parameter space
    :param metric: function of form evaluation_f(model), used to evaluate model loss
    :param distance: maximum distance in parameter space from the start point
    :param steps: at how many steps from start to end the model is evaluated
    :param normalization: normalization of direction vectors, must be one of 'filter', 'layer', 'model'
    :param deepcopy_model: indicates whether the method will deepcopy the model(s) to avoid aliasing
    :return: 1-d array of loss values along the line connecting start and end models
    """
    model_start_wrapper = wrap_model(copy.deepcopy(model) if deepcopy_model else model)

    start_point = model_start_wrapper.get_module_parameters()
    dir_one, dir_two = get_directions(normalization, start_point)

    # scale to match steps and total distance
    dir_one.mul_(((start_point.model_norm() / distance) / steps) / dir_one.model_norm())
    dir_two.mul_(((start_point.model_norm() / distance) / steps) / dir_two.model_norm())
    # Move start point so that original start params will be in the center of the plot
    dir_one.mul_(steps / 2)
    dir_two.mul_(steps / 2)
    start_point.sub_(dir_one)
    start_point.sub_(dir_two)
    dir_one.truediv_(steps / 2)
    dir_two.truediv_(steps / 2)

    data_matrix = []
    # evaluate loss in grid of (steps * steps) points, where each column signifies one step
    # along dir_one and each row signifies one step along dir_two. The implementation is again
    # a little convoluted to avoid constructive operations. Fundamentally we generate the matrix
    # [[start_point + (dir_one * i) + (dir_two * j) for j in range(steps)] for i in range(steps].
    for i in range(steps):
        data_column = []

        for j in range(steps):
            # for every other column, reverse the order in which the column is generated
            # so you can easily use in-place operations to move along dir_two
            if i % 2 == 0:
                start_point.add_(dir_two)
                data_column.append(metric(model_start_wrapper))
            else:
                start_point.sub_(dir_two)
                data_column.insert(0, metric(model_start_wrapper))

        data_matrix.append(data_column)
        start_point.add_(dir_one)

    return np.array(data_matrix)


def get_directions(normalization, start_point):
    dir_one = rand_u_like(start_point)
    dir_two = orthogonal_to(dir_one)

    if normalization == 'model':
        dir_one.model_normalize_(start_point)
        dir_two.model_normalize_(start_point)
    elif normalization == 'layer':
        dir_one.layer_normalize_(start_point)
        dir_two.layer_normalize_(start_point)
    elif normalization == 'filter':
        dir_one.filter_normalize_(start_point)
        dir_two.filter_normalize_(start_point)
    elif normalization is None:
        pass
    else:
        raise AttributeError('Unsupported normalization argument. Supported values are model, layer, and filter')
    return dir_one, dir_two
