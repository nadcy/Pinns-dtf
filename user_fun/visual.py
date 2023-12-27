import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import Checkbox, HBox, VBox, Label, Output
from IPython.display import clear_output
from IPython.display import display

def visualize_point_sets_interactive(cp_dict, xlim=(-0.5, 1.5), ylim=(-0.5, 1.5), zlim=(0, 1)):
    output = Output()

    def on_checkbox_change(change):
        if change['name'] == 'value':
            with output:
                clear_output(wait=True)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                selected_sets = {c.description: c.value for c in checkboxes}
                for name, checkbox in selected_sets.items():
                    if checkbox:
                        x, y, c = cp_dict[name]
                        ax.scatter(x, y, c, label=name)
                ax.set_title("Selected Point Sets")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Height")
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)
                ax.legend()
                plt.show()

    checkboxes = [Checkbox(value=True, description=name) for name in cp_dict.keys()]

    for c in checkboxes:
        c.observe(on_checkbox_change)

    checkbox_container = VBox([Label("Point Sets:")] + checkboxes)
    display(checkbox_container)
    display(output)

    # 初始时绘制所有选中的点集
    with output:
        on_checkbox_change({'name': 'value'})

def process_point_sets(point_sets):
    result = {}
    for name, points in point_sets.items():
        coords, colors = points
        if coords.shape[1] != 2:
            raise ValueError(f"The second dimension of 'coords' must be 2 for {name}")
        if isinstance(colors, np.ndarray):
            if colors.shape[1] == 1:
                result[f"{name}-1"] = (coords[:, 0], coords[:, 1], colors.flatten())
            else:
                for i in range(colors.shape[1]):
                    result[f"{name}-{i+1}"] = (coords[:, 0], coords[:, 1], colors[:, i])
        else:
            result[f"{name}-{colors}"] = (coords[:, 0], coords[:, 1], np.zeros_like(coords[:, 0]))
    return result


