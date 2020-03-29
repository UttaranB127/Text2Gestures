# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import bisect
import scipy.ndimage
import torch


# Spline all samples
class Spline_AS:
    def __init__(self, points, closed=True):
        assert len(points.shape) == 3
        self.closed = closed
        self.points = points.clone()
        self.tracks = {}

        # Compute arc lengths and distance track
        if closed:
            points_wrapped = torch.cat((points, points[:, 0:1]), dim=1)
        else:
            points_wrapped = points
        self.segment_lengths = torch.norm(points_wrapped[:, 1:] - points_wrapped[:, :-1], dim=-1)
        self.distances = torch.cat((torch.zeros((self.segment_lengths.shape[0], 1)).cuda().float(),
                                    torch.cumsum(self.segment_lengths, dim=1)), dim=-1)

        self._compute_tangents(self.points)

    def _compute_tangents(self, points):
        # Predefined tangent track estimated using finite differences
        der1 = torch.cat((points[:, 1:] - points[:, :-1],
                          (points[:, 0] - points[:, -1]).unsqueeze(1)), dim=1)  # 1st derivative
        if not self.closed:
            # Adjust endpoints
            der1[:, 0] = points[:, 1] - points[:, 0]
            der1[:, -1] = points[:, -1] - points[:, -2]
        tangents = der1 / (torch.norm(der1, dim=-1)[..., None] + 1e-9)  # Re-normalize
        self.tracks['tangent'] = (tangents, 'circular')

        # Predefined curvature track
        next_tangents = tangents[:, (np.arange(points.shape[1]) + 1) % points.shape[1]]
        diffs = self.tensor_angle_difference(next_tangents, tangents)
        if not self.closed:
            # Adjust endpoints
            diffs[:, 0:1] = self.tensor_angle_difference(tangents[:, 1:2], tangents[:, 0:1])
            diffs[:, -1:] = self.tensor_angle_difference(tangents[:, -1:], tangents[:, -2:-1])
        self.tracks['curvature'] = (diffs, 'linear')

    def interpolate(self, distance, track=None):
        if hasattr(distance, '__len__'):
            # Multi-point interpolation (more efficient)
            multipoint = True
            if not isinstance(distance, np.ndarray):
                distance = np.array(distance)
        else:
            multipoint = False
            distance = np.array([distance])

        if self.closed:
            # Wrap-around distance
            distance = distance % self.distances[-1]

        indices = self._get_indices(distance)
        t = ((distance - self.distances[indices]) / self.segment_lengths[indices])
        if track is None:
            result = self._lerp(self.points[indices], self.points[(indices + 1) % self.points.shape[0]],
                                t.reshape(-1, 1))
        else:
            tr = self.tracks[track]
            if len(tr[0].shape) == 2:
                t = t.reshape(-1, 1)
            p0 = tr[0][indices]
            p1 = tr[0][(indices + 1) % self.points.shape[0]]
            if tr[1] == 'linear':
                result = self._lerp(p0, p1, t)
            elif tr[1] == 'circular':
                result = self._slerp(p0, p1, t)
            else:
                raise ValueError
        if multipoint:
            return result
        else:
            return result[0]

    def add_track(self, name, data, interp_mode='linear'):
        assert self.points.shape[0] == data.shape[0]
        assert self.points.shape[1] == data.shape[1]
        assert interp_mode in ['linear', 'circular']
        self.tracks[name] = (data.clone(), interp_mode)

    def length(self):
        return self.distances[-1]

    def size(self):
        return self.points.shape[0]

    def is_closed(self):
        return self.closed

    def get_track(self, track):
        return self.tracks[track][0].clone()

    def re_parametrize(self, points_per_unit, smoothing_factor):
        """
        Re-parametrize this spline to have equal-length segments.
        """
        if self.closed:
            # Round to nearest integer
            num_steps = int(np.ceil(self.length() * points_per_unit))
            dists = np.linspace(0, self.length(), num_steps, endpoint=False)
        else:
            # Shorten the spline a little bit
            if self.length() < 1.:
                dists = np.arange(0, self.length(), step=self.length() / points_per_unit)
            else:
                dists = np.arange(0, self.length(), step=1. / points_per_unit)

        points = self.interpolate(dists)
        points_smooth = scipy.ndimage.filters.gaussian_filter1d(points.T, points_per_unit * smoothing_factor).T

        new_spline = Spline(points, closed=self.closed)
        new_spline._compute_tangents(points_smooth)
        for track_name, track_val in self.tracks.items():
            if track_name in ['tangent', 'curvature']:
                continue
            new_track = self.interpolate(dists, track=track_name)
            if track_val[1] == 'linear':
                new_track = scipy.ndimage.filters.gaussian_filter1d(new_track.T, points_per_unit * smoothing_factor).T
            new_spline.add_track(track_name, new_track, interp_mode=track_val[1])
        return new_spline

    def _get_indices(self, distance):
        # Find spline segment via binary search
        indices = np.empty(len(distance), dtype=int)
        for k, dist in enumerate(distance):
            i = bisect.bisect_right(self.distances, dist) - 1
            # These checks apply only to the non-closed spline case
            if not self.closed:
                if i < 0:
                    i = 0
                if i == len(self.points) - 1:
                    i = len(self.points) - 2
            indices[k] = i
        return indices

    @staticmethod
    def _lerp(a, b, t):
        """ Linear interpolation between two points. """
        return a + (b - a) * t

    @staticmethod
    def _slerp(a, b, t):
        """ Circular/spherical interpolation between two points. """
        magnitude = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
        o = np.arccos(np.clip(np.sum(a * b, axis=1) / magnitude, -1, 1))
        # Fall back to linear interpolation for very small angles
        # in order to avoid the degenerate case sin(x)/x -> 0/0
        mask = o < 1e-3

        # Linear interpolation
        result = np.empty(a.shape, dtype=a.dtype)
        t_ = t[mask]
        result[mask] = (1 - t_) * a[mask] + t_ * b[mask]

        # Spherical interpolation
        mask = ~mask
        t_ = t[mask].reshape(-1)
        o_ = o[mask]
        result[mask] = (np.sin((1 - t_) * o_).reshape(-1, 1) * a[mask] + np.sin(t_ * o_).reshape(-1, 1) * b[
            mask]) / np.sin(o_).reshape(-1, 1)

        return result

    @staticmethod
    def tensor_angle_difference(y, x):
        """
        Compute the signed angle difference y - x.
        Both y and x are given as (N, 2) tensors in which the last dimension
        corresponds to [cos(angle), sin(angle)].
        """
        assert len(x.shape) == 3 and len(y.shape) == 3
        assert x.shape[-1] == 2 and y.shape[-1] == 2
        # Diff = atan2(sin(y-x), cos(y-x)), where y and x are Euler angles.
        return torch.atan2(y[:, :, 1] * x[:, :, 0] - y[:, :, 0] * x[:, :, 1],
                           y[:, :, 0] * x[:, :, 0] + y[:, :, 1] * x[:, :, 1])

    @staticmethod
    def versor_angle_difference(y, x):
        """
        Compute the signed angle sum y + x.
        Both y, x, and the output are represented as versors.
        """
        cosine = y[:, :, 0] * x[:, :, 0] + y[:, :, 1] * x[:, :, 1]
        sine = y[:, :, 1] * x[:, :, 0] - y[:, :, 0] * x[:, :, 1]
        return torch.stack((cosine, sine), dim=-1)

    @staticmethod
    def versor_angle_sum(y, x):
        """
        Compute the signed angle sum y + x.
        Both y, x, and the output are represented as versors.
        """
        cosine = y[:, :, 0] * x[:, :, 0] - y[:, :, 1] * x[:, :, 1]
        sine = y[:, :, 1] * x[:, :, 0] + y[:, :, 0] * x[:, :, 1]
        return torch.stack((cosine, sine), dim=-1)

    @staticmethod
    def extract_spline_features(spline):
        spline_features = torch.cat((
            spline.get_track('curvature').unsqueeze(-1),
            spline.get_track('amplitude'),
            spline.get_track('frequency'),
            Spline_AS.versor_angle_difference(spline.get_track('direction'), spline.get_track('tangent'))
        ), dim=-1)
        return spline_features, spline.get_track('phase')


class Spline:
    def __init__(self, points, closed=True):
        assert len(points.shape) == 2
        self.closed = closed
        self.points = points.copy()
        self.tracks = {}

        # Compute arc lengths and distance track
        if closed:
            points_wrapped = np.concatenate((points, points[:1]), axis=0)
        else:
            points_wrapped = points
        self.segment_lengths = np.linalg.norm(np.diff(points_wrapped, axis=0), axis=1)
        self.distances = np.concatenate(([0], np.cumsum(self.segment_lengths, axis=0)))

        self._compute_tangents(self.points)

    def _compute_tangents(self, points):
        # Predefined tangent track estimated using finite differences
        next_points = points[(np.arange(points.shape[0]) + 1) % points.shape[0]]
        der1 = next_points - points  # 1st derivative
        if not self.closed:
            # Adjust endpoints
            der1[0] = points[1] - points[0]
            der1[-1] = points[-1] - points[-2]
        tangents = der1 / (np.linalg.norm(der1, axis=1).reshape(-1, 1) + 1e-9)  # Re-normalize
        self.tracks['tangent'] = (tangents, 'circular')

        # Predefined curvature track
        next_tangents = tangents[(np.arange(points.shape[0]) + 1) % points.shape[0]]
        diffs = self.tensor_angle_difference(next_tangents, tangents)
        if not self.closed:
            # Adjust endpoints
            diffs[0:1] = self.tensor_angle_difference(tangents[1:2], tangents[0:1])
            diffs[-1:] = self.tensor_angle_difference(tangents[-1:], tangents[-2:-1])
        self.tracks['curvature'] = (diffs, 'linear')

    def interpolate(self, distance, track=None):
        if hasattr(distance, '__len__'):
            # Multi-point interpolation (more efficient)
            multipoint = True
            if not isinstance(distance, np.ndarray):
                distance = np.array(distance)
        else:
            multipoint = False
            distance = np.array([distance])

        if self.closed:
            # Wrap-around distance
            distance = distance % self.distances[-1]

        indices = self._get_indices(distance)
        t = ((distance - self.distances[indices]) / self.segment_lengths[indices])
        if track is None:
            result = self._lerp(self.points[indices], self.points[(indices + 1) % self.points.shape[0]],
                                t.reshape(-1, 1))
        else:
            tr = self.tracks[track]
            if len(tr[0].shape) == 2:
                t = t.reshape(-1, 1)
            p0 = tr[0][indices]
            p1 = tr[0][(indices + 1) % self.points.shape[0]]
            if tr[1] == 'linear':
                result = self._lerp(p0, p1, t)
            elif tr[1] == 'circular':
                result = self._slerp(p0, p1, t)
            else:
                raise ValueError
        if multipoint:
            return result
        else:
            return result[0]

    def add_track(self, name, data, interp_mode='linear'):
        assert self.points.shape[0] == data.shape[0]
        assert interp_mode in ['linear', 'circular']
        self.tracks[name] = (data.copy(), interp_mode)

    def length(self):
        return self.distances[-1]

    def size(self):
        return self.points.shape[0]

    def is_closed(self):
        return self.closed

    def get_track(self, track):
        return self.tracks[track][0].copy()

    def re_parametrize(self, points_per_unit, smoothing_factor):
        """
        Re-parametrize this spline to have equal-length segments.
        """
        if self.closed:
            # Round to nearest integer
            num_steps = int(np.ceil(self.length() * points_per_unit))
            dists = np.linspace(0, self.length(), num_steps, endpoint=False)
        else:
            # Shorten the spline a little bit
            if self.length() < 1.:
                dists = np.arange(0, self.length(), step=self.length() / points_per_unit)
            else:
                dists = np.arange(0, self.length(), step=1. / points_per_unit)

        points = self.interpolate(dists)
        points_smooth = scipy.ndimage.filters.gaussian_filter1d(points.T, points_per_unit * smoothing_factor).T

        new_spline = Spline(points, closed=self.closed)
        new_spline._compute_tangents(points_smooth)
        for track_name, track_val in self.tracks.items():
            if track_name in ['tangent', 'curvature']:
                continue
            new_track = self.interpolate(dists, track=track_name)
            if track_val[1] == 'linear':
                new_track = scipy.ndimage.filters.gaussian_filter1d(new_track.T, points_per_unit * smoothing_factor).T
            new_spline.add_track(track_name, new_track, interp_mode=track_val[1])
        return new_spline

    def _get_indices(self, distance):
        # Find spline segment via binary search
        indices = np.empty(len(distance), dtype=int)
        for k, dist in enumerate(distance):
            i = bisect.bisect_right(self.distances, dist) - 1
            # These checks apply only to the non-closed spline case
            if not self.closed:
                if i < 0:
                    i = 0
                if i == len(self.points) - 1:
                    i = len(self.points) - 2
            indices[k] = i
        return indices

    @staticmethod
    def _lerp(a, b, t):
        """ Linear interpolation between two points. """
        return a + (b - a) * t

    @staticmethod
    def _slerp(a, b, t):
        """ Circular/spherical interpolation between two points. """
        magnitude = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
        o = np.arccos(np.clip(np.sum(a * b, axis=1) / magnitude, -1, 1))
        # Fall back to linear interpolation for very small angles
        # in order to avoid the degenerate case sin(x)/x -> 0/0
        mask = o < 1e-3

        # Linear interpolation
        result = np.empty(a.shape, dtype=a.dtype)
        t_ = t[mask]
        result[mask] = (1 - t_) * a[mask] + t_ * b[mask]

        # Spherical interpolation
        mask = ~mask
        t_ = t[mask].reshape(-1)
        o_ = o[mask]
        result[mask] = (np.sin((1 - t_) * o_).reshape(-1, 1) * a[mask] + np.sin(t_ * o_).reshape(-1, 1) * b[
            mask]) / np.sin(o_).reshape(-1, 1)

        return result

    @staticmethod
    def tensor_angle_difference(y, x):
        """
        Compute the signed angle difference y - x.
        Both y and x are given as (N, 2) tensors in which the last dimension
        corresponds to [cos(angle), sin(angle)].
        """
        assert len(x.shape) == 2 and len(y.shape) == 2
        assert x.shape[-1] == 2 and y.shape[-1] == 2
        # Diff = atan2(sin(y-x), cos(y-x)), where y and x are Euler angles.
        return np.arctan2(y[:, 1] * x[:, 0] - y[:, 0] * x[:, 1], y[:, 0] * x[:, 0] + y[:, 1] * x[:, 1])

    @staticmethod
    def versor_angle_difference(y, x):
        """
        Compute the signed angle sum y + x.
        Both y, x, and the output are represented as versors.
        """
        cosine = y[:, 0] * x[:, 0] + y[:, 1] * x[:, 1]
        sine = y[:, 1] * x[:, 0] - y[:, 0] * x[:, 1]
        return np.stack((cosine, sine), axis=1)

    @staticmethod
    def versor_angle_sum(y, x):
        """
        Compute the signed angle sum y + x.
        Both y, x, and the output are represented as versors.
        """
        cosine = y[:, 0] * x[:, 0] - y[:, 1] * x[:, 1]
        sine = y[:, 1] * x[:, 0] + y[:, 0] * x[:, 1]
        return np.stack((cosine, sine), axis=1)

    @staticmethod
    def extract_spline_features(spline):
        spline_features = np.concatenate((
            spline.get_track('curvature').reshape(-1, 1),
            spline.get_track('amplitude').reshape(-1, 1),
            spline.get_track('frequency').reshape(-1, 1),
            Spline.versor_angle_difference(spline.get_track('direction'), spline.get_track('tangent'))
        ), axis=1)
        return spline_features, spline.get_track('phase').reshape(-1, 1)
