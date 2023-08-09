import numpy as np
import torch

class array2tensor(object):
    """converts a numpy array to a torch tensor"""
        
    def __call__(self, array):
        
        ## numpy: H x W x C => torch: C x H x W
        if len(array.shape) > 3:
            array = array.transpose((0, 3, 1, 2)).astype(np.float32)
        else:
            array = array.transpose((2, 0, 1)).astype(np.float32)
        tensor = torch.from_numpy(array)
        return tensor


def tensor2array(tensor):
    """converts a torch tensor to a numpy array"""
        
    array = tensor.detach().cpu().numpy()
    
    ## torch: C x H x W => numpy: H x W x C
    if len(array.shape) > 3:
        array = array.transpose((0, 2, 3, 1)).astype(np.float32)
    else:
        array = array.transpose((1, 2, 0)).astype(np.float32)
    return array

def cartesian_to_spherical(coords):
    x, y, z = coords
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return [r, theta, phi]

def cartesian_to_polar(coords):
    x, y = coords
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return [r, theta]

def top_k(array, k):
    flat = array.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    return np.sort(np.unravel_index(indices, array.shape))

def random_moving(self, data_numpy,
                          angle_candidate=[-10., -5., 0., 5., 10.],
                          scale_candidate=[0.9, 1.0, 1.1],
                          transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                          move_time_candidate=[1]):
        
        # input: T,V,C
        data_numpy = np.transpose(data_numpy, (2, 0, 1))
        new_data_numpy = np.zeros(data_numpy.shape)
        C, T, V = data_numpy.shape
        move_time = random.choice(move_time_candidate)

        node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
        node = np.append(node, T)
        num_node = len(node)

        A = np.random.choice(angle_candidate, num_node)
        S = np.random.choice(scale_candidate, num_node)
        T_x = np.random.choice(transform_candidate, num_node)
        T_y = np.random.choice(transform_candidate, num_node)

        a = np.zeros(T)
        s = np.zeros(T)
        t_x = np.zeros(T)
        t_y = np.zeros(T)

        # linspace
        for i in range(num_node - 1):
            a[node[i]:node[i + 1]] = np.linspace(
                A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
            s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                                    node[i + 1] - node[i])
            t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                                       node[i + 1] - node[i])
            t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                                       node[i + 1] - node[i])

        theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                              [np.sin(a) * s, np.cos(a) * s]])

        # perform transformation
        for i_frame in range(T):
            xy = data_numpy[0:2, i_frame, :]
            new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))

            new_xy[0] += t_x[i_frame]
            new_xy[1] += t_y[i_frame]

            new_data_numpy[0:2, i_frame, :] = new_xy.reshape(2, V)

        new_data_numpy[2, :, :] = data_numpy[2, :, :]

        return np.transpose(new_data_numpy, (1, 2, 0))

def noise(self ,skeleton):
        
        low = -0.1
        high = -low
        # select 4 joints
        all_joint = list(range(21))
        random.shuffle(all_joint)
        selected_joint = all_joint[0:4]
        for j_id in selected_joint:
            noise_offset = np.random.uniform(low, high, 4)
            for t in range(skeleton.shape[0]):
                skeleton[t][j_id] += noise_offset
        skeleton = np.array(skeleton)
        return skeleton

def scaleInvariance(self, skeleton):

            x_c = np.copy(skeleton)

            distance = np.sqrt(np.sum((x_c[0, 1] - x_c[0, 0])**2, axis=-1))

            factor = 1/distance

            x_c *= factor

            return x_c

def translationInvariance(self, skeleton):
            # normalize by palm center value at frame=1
            skeleton_copy = np.copy(skeleton[0][1])
            skeleton -= skeleton_copy
            #skeleton = skeleton.float()
            return skeleton

def interpolate_landmarks(landmarks, L):
    l, n_landmarks, _ = landmarks.shape
    assert l > 1, "The sequence of landmarks should have at least two landmarks"

    # Compute the indices of the input landmarks
    input_indices = np.linspace(0, l - 1, l, dtype=int)

    # Compute the indices of the output landmarks
    output_indices = np.linspace(0, l - 1, L, dtype=float)

    # Compute the fractional part of the output indices
    fractions = output_indices % 1

    # Compute the integer part of the output indices
    output_indices = np.floor(output_indices).astype(int)

    # Initialize the output array
    interpolated_landmarks = np.zeros((L, n_landmarks, 6), dtype=float)

    # Compute the interpolated landmarks
    for i in range(L):
        if fractions[i] == 0:
            # The output index corresponds to an input landmark, so just copy it
            interpolated_landmarks[i] = landmarks[input_indices[output_indices[i]]]
        else:
            # Compute the mean vector between the two nearest input landmarks
            v1 = landmarks[input_indices[output_indices[i]]]
            v2 = landmarks[input_indices[min(output_indices[i] + 1, l - 1)]]
            interpolated_landmarks[i] = np.mean([v1, v2], axis=0)

    return interpolated_landmarks


def upsample(skeleton, max_frames):
    
    tensor = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(skeleton), dim=0), dim=0)


    out = nn.functional.interpolate(tensor, size=[max_frames, tensor.shape[-2] , tensor.shape[-1]], mode='trilinear')
    #tensor = torch.squeeze(torch.squeeze(out, dim=0), dim=0)
    tensor = torch.squeeze(out, dim=0)
    tensor = torch.squeeze(tensor, dim=0)
    tensor = tensor.numpy()

    return tensor

def compute_motion_features(lm):
    # Compute motion features from hand gesture landmarks
    # lm is a 3D numpy array of shape (n, 3), where n is the number of landmarks

    # Calculate the velocity vectors between each set of consecutive landmarks
    v = lm[1:] - lm[:-1]

    # Calculate the lengths of each velocity vector
    d = np.linalg.norm(v, axis=1)

    # Calculate the time differences between each pair of consecutive landmarks
    t = np.arange(len(lm))
    dt = t[1:] - t[:-1]

    # Calculate the speed and acceleration between each set of consecutive landmarks
    speed = d / dt
    accel = speed[1:] - speed[:-1]

    # Add 0 as the first speed and acceleration (since there is no previous velocity vector to compare with)
    speed = np.concatenate(([0], speed)).reshape(-1, 1)
    accel = np.concatenate(([0, 0], accel)).reshape(-1, 1)

    return speed, accel
