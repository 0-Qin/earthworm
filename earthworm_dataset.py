import os
import copy
import xml.etree.ElementTree as ET

import torch
import numpy as np
import cv2
import torch.utils.data as data


class EarthwormKeypoint(data.Dataset):
    def __init__(self,
                 img_root,
                 transforms=None,
                 anno_xml_path=None,
                 fixed_size=(256, 192)):
       
        super().__init__()
         
        # print(os.path.abspath(img_root))
        print(">>> DEBUG img_root =", img_root)
        print(">>> DEBUG abs path =", os.path.abspath(img_root))
        assert os.path.exists(img_root), f"img_root '{img_root}' not found."
        assert os.path.exists(anno_xml_path), f"anno_xml_path '{anno_xml_path}' not found."

        self.img_root = img_root
        self.anno_xml_path = anno_xml_path


        self.valid_worm_list = []
        self._parse_xml()
        self.fixed_size = fixed_size
        # self.mode = dataset
        self.transforms = transforms
        # self.coco = COCO(self.anno_path)
        # img_ids = list(sorted(self.coco.imgs.keys()))
    def _parse_xml(self):
            tree = ET.parse(self.anno_xml_path)
            root = tree.getroot()

            obj_idx = 0
            for img_elem in root.findall("image"):
                img_name = img_elem.get("name")          # e.g. "xxx/0001.png"
                width = int(img_elem.get("width"))
                height = int(img_elem.get("height"))

                img_path = os.path.join(self.img_root, img_name)
                if not os.path.exists(img_path):
                    print(f"[Warning] image file not found: {img_path}")
                    continue

                # 找到 label="centerline" 的 polyline（根据你自己的 label 名字改）
                polyline_elem = None
                for pl in img_elem.findall("polyline"):
                    if pl.get("label") == "centerline":
                        polyline_elem = pl
                        break

                if polyline_elem is None:
                    # 没有中心线标注，跳过
                    continue

                points_str = polyline_elem.get("points")
                if not points_str:
                    continue

                pts = []
                for p in points_str.split(";"):
                    p = p.strip()
                    if not p:
                        continue
                    x_str, y_str = p.split(",")
                    x, y = float(x_str), float(y_str)
                    pts.append([x, y])

                if len(pts) < 2:

                    continue

                pts = np.array(pts, dtype=np.float32)

                # ======  head / tail keypoint ======
                
                head = pts[0]      # (x_head, y_head)
                tail = pts[-1]     # (x_tail, y_tail)

                # 和 COCO 一样，用 (x, y, v) 形式，v=1 表示可见
                keypoints = np.array([
                    [head[0], head[1], 1.0],
                    [tail[0], tail[1], 1.0]
                ], dtype=np.float32)
                visible = keypoints[:, 2]

                # ====== 关键：整张图作为 bbox ======
                xmin, ymin = 0.0, 0.0
                w, h = float(width), float(height)

                info = {
                    "box": [xmin, ymin, w, h],             # 整图 bbox
                    "image_path": img_path,
                    "image_id": img_name,                  # 简单用文件名当 id
                    "image_width": width,
                    "image_height": height,
                    "obj_origin_hw": [h, w],               # 和 CocoKeypoint 类似的字段
                    "obj_index": obj_idx,
                    "score": 1.0,                          # 没有 det 分数，统一设 1.0
                    "keypoints": keypoints,                # (2, 3)
                    "visible": visible,                    # (2,)
                    "polyline": pts                        # 完整中心线点，后面玩 midline 用
                }

                self.valid_worm_list.append(info)
                print("Added:", img_name)

                obj_idx += 1

            print(f"[EarthwormKeypoint] Parsed {len(self.valid_worm_list)} worms from XML.")

    def __getitem__(self, idx):
        target = copy.deepcopy(self.valid_worm_list[idx])

        image = cv2.imread(target["image_path"])
        if image is None:
            raise RuntimeError(f"Failed to read image: {target['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            #  HRNet 那边保持一致，transforms 要同时改 image 和 keypoints
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.valid_worm_list)

    @staticmethod
    def collate_fn(batch):
        imgs_tuple, targets_tuple = tuple(zip(*batch))
        imgs_tensor = torch.stack(imgs_tuple)
        return imgs_tensor, targets_tuple


if __name__ == '__main__':
    # train = EarthwormKeypoint("/data/bsf2023/", dataset="val")
    # print(len(train))
    # t = train[0]
    # print(t)
    dataset = EarthwormKeypoint(
        img_root="code",
        anno_xml_path="code/skeletonisation_prototyping_test 2/annotations 2.xml",
        transforms=None,
        fixed_size=(256, 192)
    )
    print("Dataset size:", len(dataset))
    for i in range(5):
        img, t = dataset[i]
        print("image shape:", img.shape)
        print("box:", t["box"])
        import numpy as np
        np.set_printoptions(suppress=True)  
        print("keypoints:", t["keypoints"])

    # print("keypoints:", t["keypoints"])