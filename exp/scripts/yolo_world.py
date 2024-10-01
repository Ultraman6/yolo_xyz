from ultralytics.utils import WEIGHTS_DIR, DATASETS_DIR
from tests import SOURCE
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
from ultralytics import YOLOWorld, YOLO


def train_yolo_world():
    data = dict(
        train=dict(
            yolo_data=["Objects365.yaml"],
            grounding_data=[
                dict(
                    img_path=str(DATASETS_DIR / "flickr30k/images"),
                    json_file=str(DATASETS_DIR / "flickr30k/final_flickr_separateGT_train.json"),
                ),
                dict(
                    img_path=str(DATASETS_DIR / "GQA/images"),
                    json_file=str(DATASETS_DIR / "GQA/final_mixed_train_no_coco.json"),
                ),
            ],
        ),
        val=dict(yolo_data=["lvis.yaml"]),
    )
    model = YOLOWorld("yolov8n-worldv2.yaml")
    model.train(data=data, batch=128, epochs=100, trainer=WorldTrainerFromScratch)
    model.save(WEIGHTS_DIR / "yolov8n-worldv2.pt")

def tune_yolo_world():
    """Tests YOLO_xyz world models with CLIP support, including detection and training scenarios."""
    model = YOLO(WEIGHTS_DIR / "yolov8s-world.pt")  # no YOLOv8n-world model yet
    model.set_classes(["tree", "window"])
    model(SOURCE, conf=0.01)

    model = YOLO(WEIGHTS_DIR / "yolov8s-worldv2.pt")  # no YOLOv8n-world model yet
    # Training from a pretrained model. Eval is included at the final stage of training.
    # Use dota8.yaml which has fewer categories to reduce the inference time of CLIP model
    model.train(
        data="dota8.yaml",
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
    )

    # test WorWorldTrainerFromScratch
    from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

    model = YOLO("yolov8s-worldv2.yaml")  # no YOLOv8n-world model yet
    model.train(
        data={"train": {"yolo_data": ["dota8.yaml"]}, "val": {"yolo_data": ["dota8.yaml"]}},
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
        trainer=WorldTrainerFromScratch,
    )
    model.save(WEIGHTS_DIR / "yolov8s-worldv2.pt")

# 导出操作暂时只能就地
def _test_for_onnx():
    # 加载训练好的模型
    # model = YOLOWorld("yolov10n-worldv2.yaml")
    model = YOLOWorld("yolov10n-worldv2.yaml")
    # 将模型转为onnx格式
    success = model.export(format='onnx')
    print(success)


if __name__ == '__main__':
    train_yolo_world()