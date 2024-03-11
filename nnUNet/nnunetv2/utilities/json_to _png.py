import json
import os
import os.path as osp
from PIL import Image, ImageDraw
import PIL.Image
import yaml
from labelme import utils
import base64

def json_to_png(json_path, output_folder):
    # 读取JSON文件
    with open(json_path, 'r') as file:
        data = json.load(file)

    # 解析JSON数据并绘制PNG图片
    image = Image.new('RGB', (1920, 1080), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for shape in data['shapes']:
        points = shape['points']
        color = shape.get('color', (0, 0, 0))  # 使用默认颜色 (0, 0, 0) 如果 color 字段不存在
        draw.polygon(points, fill=color)

    # 保存PNG文件
    file_name = json_path.split('/')[-1].split('.')[0]
    output_path = f"{output_folder}/{file_name}.png"
    image.save(output_path, 'PNG')



# label.png
# 401.png
if __name__ == '__main__':
    # main()

    # 输入和输出文件夹路径
    input_folder = '/media/turenzhe/黎柏宏3U/data_raw/test_label'
    out_folder = '/media/turenzhe/黎柏宏3U/data_converted/data_test01/test/mask'

    # 获取输入文件夹中所有JSON文件的路径
    json_files = [file for file in os.listdir(input_folder) if file.endswith('.json')]

    # 遍历每个JSON文件并进行转换
    for json_file in json_files:
        json_path = os.path.join(input_folder, json_file)
        # json_to_png(json_path, output_folder)
        data = json.load(open(json_path))

        if data['imageData']:
            imageData = data['imageData']
        else:
            imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')
        # print(imageData)
        img = utils.img_b64_to_arr(imageData)
        # print(img)
        # exit()
        label_name_to_value = {'_background_': 0,'LGP':1, 'LHV':2}
        for shape in data['shapes']:
            label_name = shape['label'] #lgp lhv
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value

        # label_values must be dense
        label_values, label_names = [], []
        for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
            label_values.append(lv)
            label_names.append(ln)
        assert label_values == list(range(len(label_values)))

        lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

        captions = ['{}: {}'.format(lv, ln)
                    for ln, lv in label_name_to_value.items()]
        lbl_viz = utils.draw_label(lbl, img, captions)

        out_dir = osp.basename(json_file).replace('json', 'png')
        # out_dir = osp.join(osp.dirname(count[i]), out_dir)
        output_folder = osp.join(out_folder, out_dir)
        # if not osp.exists(output_folder):
        #     os.mkdir(output_folder)
        #
        # PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
        # # PIL.Image.fromarray(lbl).save(osp.join(out_dir, 'label.png'))
        # utils.lblsave(osp.join(output_folder, 'label.png'), lbl)
        utils.lblsave(output_folder, lbl)

        print('Saved to: %s' % output_folder)
