
import os
import shutil
import json
# import glob


def label_correct(path, count=None, copy=False, display=True):
    fs = next(os.walk(path))[2]
    count = count or 0
    for f in fs:
        correct = None
        f_path = os.path.join(path, f)
        if f.split(".")[-1] == "json":
            with open(f_path, 'r') as fp:
                data = json.load(fp)
                # print(len(data['shapes']))
                for i in range(len(data['shapes'])):
                    if data['shapes'][i]["shape_type"] not in ["polygon", "linestrip"]:
                        wrong_shape = data['shapes'][i]["shape_type"]
                        count += 1
                        correct = True
                        break
                    else:
                        continue
                if correct:
                    # file_name = ".".join([f.split(".")[0], "png"])
                    file_name = f
                    if display:
                        print("Image: {} Wrong type: {}".format(f,
                                                                wrong_shape))
                    # print(file_name)
                    file_path = os.path.join(path, file_name)
                    # print(file_path)
                    target_path = "/".join(path.split('/')[:-1] +
                                           [path.split('/')[-1] + "_wrong"])
                    # print(target_path)
                    if copy:
                        if not os.path.exists(target_path):
                            os.makedirs(target_path)
                        shutil.copy(file_path, target_path)
                    # break
        else:
            continue

    print("\n=== Image number: {} in {} ===".format(count, path))
    return count


num = label_correct("../datasets/test111")
num = label_correct("../datasets/train111", num)
label_correct("../datasets/val111", num)
