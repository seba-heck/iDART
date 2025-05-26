import numpy as np
import argparse

out_path = "tmp.obj"

def sample_point(n=1,h_size=(0,2),r_size=(0,0.3),a_size=(0,2*3.1415),pos=(0,0,0)):
    h = np.random.rand(n)* (h_size[1] - h_size[0]) + h_size[0]
    r = np.random.rand(n)* (r_size[1] - r_size[0]) + r_size[0]
    a = np.random.rand(n)* (a_size[1] - a_size[0]) + a_size[0]

    x = r * np.cos(a)
    y = r * np.sin(a)

    out = np.zeros((n,3))
    out[:,0] = x + pos[0]
    out[:,1] = y + pos[1]
    out[:,2] = h + pos[2]
    return out.reshape(n,3)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000, help='Number of points.')
    parser.add_argument('--save', type=bool, default=False, help="Print to terminal or save it as 'tmp.obj' file.")
    args = parser.parse_args()

    points = sample_point(args.n, pos=(1,-1,0))

    points[:, [0,1,2]] = points[:, [0,2,1]]
    points[:, 2] = -points[:, 2]

    output = ""
    for [x,y,z] in points:
        output += f"v {x} {y} {z}\n"

    if args.save:
        with open(out_path, 'w') as fout:
            fout.write(output)
    else:
        print(output)