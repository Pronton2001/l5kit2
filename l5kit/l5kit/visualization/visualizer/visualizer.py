from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set

import bokeh.io
import bokeh.plotting
import numpy as np
from bokeh.layouts import column, LayoutDOM, row, gridplot
from bokeh.models import CustomJS, HoverTool, Slider, Button, TapTool
from bokeh.plotting import ColumnDataSource
from bokeh.io import curdoc

from bokeh.io import output_notebook, show



from l5kit.visualization.visualizer.common import (AgentVisualization, CWVisualization, EgoVisualization,
                                                   FrameVisualization, LaneVisualization, TrajectoryVisualization)


def _visualization_list_to_dict(visualisation_list: List[Any], null_el: Any) -> Dict[str, Any]:
    """Convert a list of NamedTuple into a dict, where:
    - the NamedTuple fields are the dict keys;
    - the dict value are lists;

    :param visualisation_list: a list of NamedTuple
    :param null_el: an element to be used as null if the list is empty (it can crash visualisation)
    :return: a dict with the same information
    """
    visualisation_list = visualisation_list if len(visualisation_list) else [null_el]
    visualisation_dict: DefaultDict[str, Any] = defaultdict(list)

    keys_set: Set[str] = set(visualisation_list[0]._asdict().keys())
    for el in visualisation_list:
        for k, v in el._asdict().items():
            if k not in keys_set:
                raise ValueError("keys set is not consistent between elements in the list")
            visualisation_dict[k].append(v)
    return dict(visualisation_dict)

def visualize(scene_index: int, frames: List[FrameVisualization]) -> LayoutDOM:
    """Visualise a scene using Bokeh.

    :param scene_index: the index of the scene, used only as the title
    :param frames: a list of FrameVisualization objects (one per frame of the scene)
    """

    agent_hover = HoverTool(
        mode="mouse",
        names=["agents"],
        tooltips=[
            ("Type", "@agent_type"),
            ("Probability", "@prob{0.00}%"),
            ("Track id", "@track_id"),
        ],
    )
    out: List[Dict[str, ColumnDataSource]] = []

    trajectories_labels = np.unique([traj.legend_label for frame in frames for traj in frame.trajectories])

    for frame_idx, frame in enumerate(frames):
        # we need to ensure we have something otherwise js crashes
        ego_dict = _visualization_list_to_dict([frame.ego], EgoVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                             color="black", center_x=0,
                                                                             center_y=0))

        agents_dict = _visualization_list_to_dict(frame.agents, AgentVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                   color="black", track_id=-2,
                                                                                   agent_type="", prob=0.))

        lanes_dict = _visualization_list_to_dict(frame.lanes, LaneVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                color="black"))

        crosswalk_dict = _visualization_list_to_dict(frame.crosswalks, CWVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                       color="black"))

        # for trajectory we extract the labels so that we can show them in the legend
        trajectory_dict: Dict[str, Dict[str, Any]] = {}
        for trajectory_label in trajectories_labels:
            trajectories = [el for el in frame.trajectories if el.legend_label == trajectory_label]
            trajectory_dict[trajectory_label] = _visualization_list_to_dict(trajectories,
                                                                            TrajectoryVisualization(xs=np.empty(0),
                                                                                                    ys=np.empty(0),
                                                                                                    color="black",
                                                                                                    legend_label="none",
                                                                                                    track_id=-2))

        frame_dict = dict(ego=ColumnDataSource(ego_dict), agents=ColumnDataSource(agents_dict),
                          lanes=ColumnDataSource(lanes_dict),
                          crosswalks=ColumnDataSource(crosswalk_dict))
        frame_dict.update({k: ColumnDataSource(v) for k, v in trajectory_dict.items()})

        out.append(frame_dict)

    f = bokeh.plotting.figure(
        title="Scene {}".format(scene_index),
        match_aspect=True,
        x_range=(out[0]["ego"].data["center_x"][0] - 50, out[0]["ego"].data["center_x"][0] + 50),
        y_range=(out[0]["ego"].data["center_y"][0] - 50, out[0]["ego"].data["center_y"][0] + 50),
        tools=["pan", "wheel_zoom", agent_hover, "save", "reset"],
        active_scroll="wheel_zoom",
    )

    f.xgrid.grid_line_color = None
    f.ygrid.grid_line_color = None

    f.patches(line_width=0, alpha=0.5, color="color", source=out[0]["lanes"])
    f.patches(line_width=0, alpha=0.5, color="#B5B50D", source=out[0]["crosswalks"])
    f.patches(line_width=2, color="#B53331", source=out[0]["ego"])
    f.patches(line_width=2, color="color", name="agents", source=out[0]["agents"])

    js_string = """
            sources["lanes"].data = frames[cb_obj.value]["lanes"].data;
            sources["crosswalks"].data = frames[cb_obj.value]["crosswalks"].data;
            sources["agents"].data = frames[cb_obj.value]["agents"].data;
            sources["ego"].data = frames[cb_obj.value]["ego"].data;

            var center_x = frames[cb_obj.value]["ego"].data["center_x"][0];
            var center_y = frames[cb_obj.value]["ego"].data["center_y"][0];

            figure.x_range.setv({"start": center_x-50, "end": center_x+50})
            figure.y_range.setv({"start": center_y-50, "end": center_y+50})

            sources["lanes"].change.emit();
            sources["crosswalks"].change.emit();
            sources["agents"].change.emit();
            sources["ego"].change.emit();
        """

    for trajectory_name in trajectories_labels:
        f.multi_line(alpha=0.8, line_width=3, source=out[0][trajectory_name], color="color",
                     legend_label=trajectory_name)
        js_string += f'sources["{trajectory_name}"].data = frames[cb_obj.value]["{trajectory_name}"].data;\n' \
                     f'sources["{trajectory_name}"].change.emit();\n'

    slider_callback = CustomJS(
        args=dict(figure=f, sources=out[0], frames=out),
        code=js_string,
    )

    slider = Slider(start=0, end=len(frames), value=0, step=1, title="frame")
    slider.js_on_change("value", slider_callback)

    f.legend.location = "top_left"
    f.legend.click_policy = "hide"

    layout = column(f, slider)
    return layout


def visualize2(scene_index: int, frames: List[FrameVisualization]) -> LayoutDOM:
    """Visualise a scene using Bokeh.

    :param scene_index: the index of the scene, used only as the title
    :param frames: a list of FrameVisualization objects (one per frame of the scene)
    """

    agent_hover = HoverTool(
        mode="mouse",
        names=["agents"],
        tooltips=[
            ("Type", "@agent_type"),
            ("Probability", "@prob{0.00}%"),
            ("Track id", "@track_id"),
        ],
    )
    out: List[Dict[str, ColumnDataSource]] = []

    trajectories_labels = np.unique([traj.legend_label for frame in frames for traj in frame.trajectories])

    for frame_idx, frame in enumerate(frames):
        # we need to ensure we have something otherwise js crashes
        ego_dict = _visualization_list_to_dict([frame.ego], EgoVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                             color="black", center_x=0,
                                                                             center_y=0))

        agents_dict = _visualization_list_to_dict(frame.agents, AgentVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                   color="black", track_id=-2,
                                                                                   agent_type="", prob=0.))

        lanes_dict = _visualization_list_to_dict(frame.lanes, LaneVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                color="black"))

        crosswalk_dict = _visualization_list_to_dict(frame.crosswalks, CWVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                       color="black"))

        # for trajectory we extract the labels so that we can show them in the legend
        trajectory_dict: Dict[str, Dict[str, Any]] = {}
        for trajectory_label in trajectories_labels:
            trajectories = [el for el in frame.trajectories if el.legend_label == trajectory_label]
            trajectory_dict[trajectory_label] = _visualization_list_to_dict(trajectories,
                                                                            TrajectoryVisualization(xs=np.empty(0),
                                                                                                    ys=np.empty(0),
                                                                                                    color="black",
                                                                                                    legend_label="none",
                                                                                                    track_id=-2))

        frame_dict = dict(ego=ColumnDataSource(ego_dict), agents=ColumnDataSource(agents_dict),
                          lanes=ColumnDataSource(lanes_dict),
                          crosswalks=ColumnDataSource(crosswalk_dict))
        frame_dict.update({k: ColumnDataSource(v) for k, v in trajectory_dict.items()})

        out.append(frame_dict)

    f = bokeh.plotting.figure(
        title="Scene {}".format(scene_index),
        match_aspect=True,
        x_range=(out[0]["ego"].data["center_x"][0] - 50, out[0]["ego"].data["center_x"][0] + 50),
        y_range=(out[0]["ego"].data["center_y"][0] - 50, out[0]["ego"].data["center_y"][0] + 50),
        tools=["pan", "wheel_zoom", agent_hover, "save", "reset"],
        active_scroll="wheel_zoom",
    )

    f.xgrid.grid_line_color = None
    f.ygrid.grid_line_color = None

    f.patches(line_width=0, alpha=0.5, color="color", source=out[0]["lanes"])
    f.patches(line_width=0, alpha=0.5, color="#B5B50D", source=out[0]["crosswalks"])
    f.patches(line_width=2, color="#B53331", source=out[0]["ego"])
    f.patches(line_width=2, color="color", name="agents", source=out[0]["agents"])

    js_string = """
            sources["lanes"].data = frames[cb_obj.value]["lanes"].data;
            sources["crosswalks"].data = frames[cb_obj.value]["crosswalks"].data;
            sources["agents"].data = frames[cb_obj.value]["agents"].data;
            sources["ego"].data = frames[cb_obj.value]["ego"].data;

            var center_x = frames[cb_obj.value]["ego"].data["center_x"][0];
            var center_y = frames[cb_obj.value]["ego"].data["center_y"][0];

            figure.x_range.setv({"start": center_x-50, "end": center_x+50})
            figure.y_range.setv({"start": center_y-50, "end": center_y+50})

            sources["lanes"].change.emit();
            sources["crosswalks"].change.emit();
            sources["agents"].change.emit();
            sources["ego"].change.emit();
        """

    for trajectory_name in trajectories_labels:
        f.multi_line(alpha=0.8, line_width=3, source=out[0][trajectory_name], color="color",
                     legend_label=trajectory_name)
        js_string += f'sources["{trajectory_name}"].data = frames[cb_obj.value]["{trajectory_name}"].data;\n' \
                     f'sources["{trajectory_name}"].change.emit();\n'

    slider_callback = CustomJS(
        args=dict(figure=f, sources=out[0], frames=out),
        code=js_string,
    )

    slider = Slider(start=0, end=len(frames), value=0, step=1, title="frame")
    button = Button(label="Export", button_type="success")
    ## Define Widgets

    ## Define Callbacks
    # curr_frame = 0
    # def update_chart():
    #     global curr_frame
    #     curr_frame += 1
    #     if curr_frame == 30:
    #         curr_frame = 0

    #     raise ValueError(slider.value)
    #     slider.value = curr_frame
    #     # intermediate_df = population[[str(curr_year)]].reset_index().rename(columns={str(curr_year): "Population"})
    #     # intermediate_df["size"] = intermediate_df["Population"] / 5e6
    #     # points.data_source.data = intermediate_df

    #     # label.text = "Year: {}".format(curr_year)

    def button_callback():
        # get the current frame index
        current_frame_index = slider.value

        # update the frame index
        next_frame_index = current_frame_index + 1 if current_frame_index < len(frames) - 1 else 0
        slider.value = next_frame_index# callback = None
    # def execute_animation():
    #     global callback
    #     # if btn.label == "Play":
    #     #     btn.label = "Pause"
    #     callback = curdoc().add_periodic_callback(update_chart, 100)
    #     # else:
    #     #     btn.label = "Play"
    #     #     curdoc().remove_periodic_callback(callback)


    f.legend.location = "top_left"
    f.legend.click_policy = "hide"
    # curdoc().add_periodic_callback(update_chart, 100)

    slider.js_on_change("value", slider_callback)
    button.on_click(button_callback)
    layout = column(f, button, slider)

    return layout

def visualize3(scene_index: int, frames: List[FrameVisualization]) -> LayoutDOM:
    """Visualise a scene using Bokeh.

    :param scene_index: the index of the scene, used only as the title
    :param frames: a list of FrameVisualization objects (one per frame of the scene)
    """

    agent_hover = HoverTool(
        mode="mouse",
        names=["agents"],
        tooltips=[
            ("Type", "@agent_type"),
            ("Probability", "@prob{0.00}%"),
            ("Track id", "@track_id"),
        ],
    )
    out: List[Dict[str, ColumnDataSource]] = []

    trajectories_labels = np.unique([traj.legend_label for frame in frames for traj in frame.trajectories])

    for frame_idx, frame in enumerate(frames):
        # we need to ensure we have something otherwise js crashes
        ego_dict = _visualization_list_to_dict([frame.ego], EgoVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                             color="black", center_x=0,
                                                                             center_y=0))

        agents_dict = _visualization_list_to_dict(frame.agents, AgentVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                   color="black", track_id=-2,
                                                                                   agent_type="", prob=0.))

        lanes_dict = _visualization_list_to_dict(frame.lanes, LaneVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                color="black"))

        crosswalk_dict = _visualization_list_to_dict(frame.crosswalks, CWVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                       color="black"))

        # for trajectory we extract the labels so that we can show them in the legend
        trajectory_dict: Dict[str, Dict[str, Any]] = {}
        for trajectory_label in trajectories_labels:
            trajectories = [el for el in frame.trajectories if el.legend_label == trajectory_label]
            trajectory_dict[trajectory_label] = _visualization_list_to_dict(trajectories,
                                                                            TrajectoryVisualization(xs=np.empty(0),
                                                                                                    ys=np.empty(0),
                                                                                                    color="black",
                                                                                                    legend_label="none",
                                                                                                    track_id=-2))

        frame_dict = dict(ego=ColumnDataSource(ego_dict), agents=ColumnDataSource(agents_dict),
                          lanes=ColumnDataSource(lanes_dict),
                          crosswalks=ColumnDataSource(crosswalk_dict))
        frame_dict.update({k: ColumnDataSource(v) for k, v in trajectory_dict.items()})

        out.append(frame_dict)

    f = bokeh.plotting.figure(
        title="Scene {}".format(scene_index),
        match_aspect=True,
        x_range=(out[0]["ego"].data["center_x"][0] - 50, out[0]["ego"].data["center_x"][0] + 50),
        y_range=(out[0]["ego"].data["center_y"][0] - 50, out[0]["ego"].data["center_y"][0] + 50),
        tools=["pan", "wheel_zoom", agent_hover, "save", "reset"],
        active_scroll="wheel_zoom",
    )

    f.xgrid.grid_line_color = None
    f.ygrid.grid_line_color = None

    f.patches(line_width=0, alpha=0.5, color="color", source=out[0]["lanes"])
    f.patches(line_width=0, alpha=0.5, color="#B5B50D", source=out[0]["crosswalks"])
    f.patches(line_width=2, color="#B53331", source=out[0]["ego"])
    f.patches(line_width=2, color="color", name="agents", source=out[0]["agents"])

    button = Button(label="Play", button_type="success")
    # button = Button(label="Left", button_type="success")
    # button = Button(label="Right", button_type="success")

    js_string = """
                var counter = 0;
                var intervalId = setInterval(function() {
                    sources["lanes"].data = frames[counter]["lanes"].data;
                    sources["crosswalks"].data = frames[counter]["crosswalks"].data;
                    sources["agents"].data = frames[counter]["agents"].data;
                    sources["ego"].data = frames[counter]["ego"].data;

                    var center_x = frames[counter]["ego"].data["center_x"][0];
                    var center_y = frames[counter]["ego"].data["center_y"][0];

                    figure.x_range.setv({"start": center_x-50, "end": center_x+50})
                    figure.y_range.setv({"start": center_y-50, "end": center_y+50})

                    sources["lanes"].change.emit();
                    sources["crosswalks"].change.emit();
                    sources["agents"].change.emit();
                    sources["ego"].change.emit();
                    counter +=1;
                    
            """
    for trajectory_name in trajectories_labels:
        f.multi_line(alpha=0.8, line_width=3, source=out[0][trajectory_name], color="color",
                     legend_label=trajectory_name)
        js_string += f'sources["{trajectory_name}"].data = frames[counter]["{trajectory_name}"].data;\n' \
                     f'sources["{trajectory_name}"].change.emit();\n'
    js_string+='''
                    if (counter >= frames.length - 1) {
                        counter = 0;
                    }
                }, 100); // 100, change as needed
                '''

    button_callback = CustomJS(
        args=dict(figure=f, sources=out[0], frames=out),
        code = js_string
    )

    # slider_callback = CustomJS(
    #     args=dict(figure=f, sources=out[0], frames=out),
    #     code=js_string,
    # )

    # slider = Slider(start=0, end=len(frames), value=0, step=1, title="frame")

   
    f.legend.location = "top_left"
    f.legend.click_policy = "hide"


    # slider.js_on_change("value", slider_callback)
    button.js_on_click(button_callback)

    

    # Display the buttons and the current images

    # show(row(left_button, right_button, cannot_tell_button, same_button, button, f))

    return f, button
    # layout = column(f,button)
    # return layout

idx1 = 1
idx2 = 1
def visualize4(scene_index: int, frames: List[FrameVisualization], doc, trajectory) -> LayoutDOM:
    """Visualise a scene using Bokeh.

    :param scene_index: the index of the scene, used only as the title
    :param frames: a list of FrameVisualization objects (one per frame of the scene)
    """

    agent_hover = HoverTool(
        mode="mouse",
        names=["agents"],
        tooltips=[
            ("Type", "@agent_type"),
            ("Probability", "@prob{0.00}%"),
            ("Track id", "@track_id"),
        ],
    )
    out: List[Dict[str, ColumnDataSource]] = []

    trajectories_labels = np.unique([traj.legend_label for frame in frames for traj in frame.trajectories])

    for frame_idx, frame in enumerate(frames):
        # we need to ensure we have something otherwise js crashes
        ego_dict = _visualization_list_to_dict([frame.ego], EgoVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                             color="black", center_x=0,
                                                                             center_y=0))

        agents_dict = _visualization_list_to_dict(frame.agents, AgentVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                   color="black", track_id=-2,
                                                                                   agent_type="", prob=0.))

        lanes_dict = _visualization_list_to_dict(frame.lanes, LaneVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                color="black"))

        crosswalk_dict = _visualization_list_to_dict(frame.crosswalks, CWVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                       color="black"))

        # for trajectory we extract the labels so that we can show them in the legend
        trajectory_dict: Dict[str, Dict[str, Any]] = {}
        for trajectory_label in trajectories_labels:
            trajectories = [el for el in frame.trajectories if el.legend_label == trajectory_label]
            trajectory_dict[trajectory_label] = _visualization_list_to_dict(trajectories,
                                                                            TrajectoryVisualization(xs=np.empty(0),
                                                                                                    ys=np.empty(0),
                                                                                                    color="black",
                                                                                                    legend_label="none",
                                                                                                    track_id=-2))

        frame_dict = dict(ego=ColumnDataSource(ego_dict), agents=ColumnDataSource(agents_dict),
                          lanes=ColumnDataSource(lanes_dict),
                          crosswalks=ColumnDataSource(crosswalk_dict))
        frame_dict.update({k: ColumnDataSource(v) for k, v in trajectory_dict.items()})

        out.append(frame_dict)

    f = bokeh.plotting.figure(
        title="Scene {}".format(scene_index),
        match_aspect=True,
        x_range=(out[0]["ego"].data["center_x"][0] - 50, out[0]["ego"].data["center_x"][0] + 50),
        y_range=(out[0]["ego"].data["center_y"][0] - 50, out[0]["ego"].data["center_y"][0] + 50),
        tools=["pan", "wheel_zoom", agent_hover, "save", "reset"],
        active_scroll="wheel_zoom",
    )

    f.xgrid.grid_line_color = None
    f.ygrid.grid_line_color = None

    f.patches(line_width=0, alpha=0.5, color="color", source=out[0]["lanes"])
    f.patches(line_width=0, alpha=0.5, color="#B5B50D", source=out[0]["crosswalks"])
    f.patches(line_width=2, color="#B53331", source=out[0]["ego"])
    f.patches(line_width=2, color="color", name="agents", source=out[0]["agents"])

    

    # Create a list of sources to update
    # sources = [out[0]["lanes"], out[0]["crosswalks"], out[0]["ego"], out[0]["agents"]]
    # sources.extend([out[0][k] for k in trajectories_labels])

    # Update the sources for each frame and append the figure to the list of figures
    if trajectory == 'left':
        def update_chart():
            global idx1
            frame = out[idx1]
            for k, v in frame.items():
                if k in out[0]:
                    out[0][k].data.update(v.data)

            idx1 +=1
            if idx1 >= len(frames) - 1:
                idx1 = 0
        doc.add_periodic_callback(update_chart, 100)
    else: # right
        def update_chart2():
            global idx2
            frame = out[idx2]
            for k, v in frame.items():
                if k in out[0]:
                    out[0][k].data.update(v.data)

            idx2 +=1
            if idx2 >= len(frames) - 1:
                idx2 = 0
        doc.add_periodic_callback(update_chart2, 100)
    return f
