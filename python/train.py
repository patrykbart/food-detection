import argparse

from model import YOLOModel
from data import load_dataset


def main(params):

    print("\n" + "#" * 25 + "\nLoad Dataset\n" + "#" * 25 + "\n")
    dataset = load_dataset(
        path=params.dataset,
        grid_size=params.grid,
        classes=params.classes,
        b_boxes=params.b_boxes,
        batch=params.batch
    )

    model = YOLOModel(
        height=224,
        width=224,
        channels=3,
        classes=params.classes,
        b_boxes=params.b_boxes,
        grid=params.grid,
        batch=params.batch
    )

    model.build()
    model.compile()

    print("\n" + "#" * 25 + "\nFit Model\n" + "#" * 25 + "\n")
    model.train(
        dataset=dataset,
        epochs=params.epochs
    )

    print("\n" + "#" * 25 + "\nSave Model\n" + "#" * 25 + "\n")
    model.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--b_boxes", type=int, default=1)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=800)

    args = parser.parse_args()
    main(args)
