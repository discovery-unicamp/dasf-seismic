#!/usr/bin/env python3

import dask.array as da
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import xarray
from dasf.transforms.base import Operator


def make_box_layout():
     return widgets.Layout(
        padding='5px 5px 5px 5px'
     )


class PlotSEGYDistributedInternal(widgets.VBox):
    def __init__(self, data, swapaxes=None, figsize=(11, 8)):
        super().__init__()
        self.output = widgets.Output()
        
        self.contour = 0
        
        self.data = data

        self.swapaxes = swapaxes
 
        with self.output:
            if isinstance(figsize, tuple) and len(figsize) == 2:
                fig = plt.figure(figsize=figsize)
            
            self.ax = fig.gca()
       
        if isinstance(self.data, xarray.Dataset):
            self.iidx = self.data.iline[0]
            self.xidx = self.data.xline[0]
            self.zidx = self.data.twt[0]

            self.iline = self.data.sel(iline=self.iidx).transpose("twt", "xline").data
            self.xline = self.data.sel(xline=self.xidx).transpose("twt", "iline").data
            self.zslice = self.data.sel(twt=self.zidx, method="nearest").transpose("iline", "xline").data

            plot_img = self.iline.values
        elif isinstance(self.data, np.ndarray) or isinstance(self.data, da.core.Array):
            self.iidx = 0
            self.xidx = 0
            self.zidx = 0

            self.iline = self.data[self.iidx, :, :]
            self.xline = self.data[:, self.xidx, :]
            self.zslice = self.data[:, :, self.zidx]

            plot_img = self.iline

        imshow_kwargs = dict(
            cmap="gray", aspect="auto", interpolation="bicubic"
        )

        if isinstance(self.data, xarray.Dataset):
            min_iline = self.data.iline[0]
            max_iline = self.data.iline[-1]
            step_iline = self.data.iline[1] - self.data.iline[0]
            min_xline = self.data.xline[0]
            max_xline = self.data.xline[-1]
            step_xline = self.data.xline[1] - self.data.xline[0]
            min_twt = self.data.twt[0]
            max_twt = self.data.twt[-1]
            step_twt = self.data.twt[1] - self.data.twt[0]
        elif isinstance(self.data, np.ndarray) or isinstance(self.data, da.core.Array):
            min_iline = 0
            max_iline = self.data.shape[0]
            step_iline = 1
            min_xline = 0
            max_xline = self.data.shape[1]
            step_xline = 1
            min_twt = 0
            max_twt = self.data.shape[2]
            step_twt = 1

        # define widgets
        self.iline_slider = widgets.IntSlider(
            value=self.iidx,
            min=min_iline,
            max=max_iline,
            step=step_iline,
            description='iline:',
            continuous_update=False
        )
        self.xline_slider = widgets.IntSlider(
            value=self.xidx,
            min=min_xline,
            max=max_xline,
            step=step_xline,
            description='xline:',
            continuous_update=False
        )
        self.zslice_slider = widgets.IntSlider(
            value=self.zidx,
            min=min_twt,
            max=max_twt,
            step=step_twt,
            description='zslice:',
            continuous_update=False
        )
        
        self.dropdown = widgets.Dropdown(
            value='iline', 
            options=['iline', 'xline', 'zslice'], 
            description='Type:'
        )

        if isinstance(self.data, xarray.Dataset):
            plot_type_options = ['raw', 'contour', 'contourf']
        elif isinstance(self.data, np.ndarray) or isinstance(self.data, da.core.Array):
            plot_type_options = ['raw']
        
        self.plot_type = widgets.RadioButtons(
            options=plot_type_options,
            description='Plot type:',
            disabled=False
        )
        
        self.colors_dropdown = widgets.Dropdown(
            value='gray', 
            options=['gray', 'seismic', 'rainbow', 'coolwarm'], 
            description='Plot color:'
        )
        
        self.cmap = 'gray'        
 
        main_controls = widgets.VBox([
            self.iline_slider,
            self.xline_slider,
            self.zslice_slider,
            self.dropdown
        ])
        main_controls.layout = make_box_layout()
        
        plot_controls = widgets.VBox([
            self.plot_type,
            self.colors_dropdown
        ])
        plot_controls.layout = make_box_layout()
        
        controls = widgets.HBox([
            main_controls,
            plot_controls
        ])
         
        out_box = widgets.Box([self.output])
        self.output.layout = make_box_layout()
 
        # observe stuff
        self.iline_slider.observe(self.update_iline, 'value')
        self.xline_slider.observe(self.update_xline, 'value')
        self.zslice_slider.observe(self.update_zslice, 'value')
        self.dropdown.observe(self.update_dropdown, 'value')
        self.plot_type.observe(self.update_plot_type, 'value')
        self.colors_dropdown.observe(self.update_colors_dropdown, 'value')
         
        # add to children
        self.children = [controls, self.output]
        
        self.slider_color = self.iline_slider.style.handle_color
        
        self.update_dropdown(None)

        if isinstance(self.data, xarray.Dataset):
            plot_data = self.data.sel(iline=self.data.iline[0]).transpose("twt", "xline").data
            plot_iline_label = "iline=" + str(self.data.iline[0].data)
        elif isinstance(self.data, np.ndarray) or isinstance(self.data, da.core.Array):
            plot_data = self.data[0, :, :]
            plot_iline_label = "iline=0"
     
    def update(self, default_plot, title):
        """Draw line in plot"""
        self.ax.clear()
        
        imshow_kwargs = dict(
            cmap=self.cmap, aspect="auto", interpolation="bicubic",
        )
        
        self.ax.set_title(title)
        
        
        if self.contour:
            if self.contour == 1:
                default_plot.plot.contour(yincrease=False, add_colorbar=False)
            elif self.contour == 2:
                default_plot.plot.contourf(yincrease=False, add_colorbar=False)
            #self.ax.contour(contour)
        else:
            if isinstance(self.data, xarray.Dataset):
                plot_img = default_plot.values
            elif isinstance(self.data, np.ndarray) or isinstance(self.data, da.core.Array):
                plot_img = default_plot

            if self.swapaxes is not None and isinstance(self.swapaxes, tuple) and len(self.swapaxes) == 2:
                plot_img = np.swapaxes(plot_img, self.swapaxes[0], self.swapaxes[1])

            self.ax.imshow(plot_img, **imshow_kwargs)

        with self.output:
            self.output.clear_output(wait=True)
 
    def update_iline(self, change):
        if not change:
            if isinstance(self.data, xarray.Dataset):
                self.iline = self.data.sel(iline=self.data.iline[0]).transpose("twt", "xline").data
                title = "iline=" + str(self.data.iline[0].data)
            elif isinstance(self.data, np.ndarray) or isinstance(self.data, da.core.Array):
                self.iline = self.data[0, :, :]
                title = "iline=0"
        else:
            if isinstance(self.data, xarray.Dataset):
                self.iline = self.data.sel(iline=change.new).transpose("twt", "xline").data
            elif isinstance(self.data, np.ndarray) or isinstance(self.data, da.core.Array):
                self.iline = self.data[change.new, :, :]
            title = "iline=" + str(change.new)
        self.update(self.iline, title)
 
    def update_xline(self, change):
        if not change:
            if isinstance(self.data, xarray.Dataset):
                self.xline = self.data.sel(xline=self.data.xline[0]).transpose("twt", "iline").data
                title = "xline=" + str(self.data.xline[0].data)
            elif isinstance(self.data, np.ndarray) or isinstance(self.data, da.core.Array):
                self.xline = self.data[:, 0, :]
                title = "xline=0"
        else:
            if isinstance(self.data, xarray.Dataset):
                self.xline = self.data.sel(xline=change.new).transpose("twt", "iline").data
            elif isinstance(self.data, np.ndarray) or isinstance(self.data, da.core.Array):
                self.xline = self.data[:, change.new, :]
            title = "xline=" + str(change.new)
        self.update(self.xline, title)
 
    def update_zslice(self, change):
        if not change:
            if isinstance(self.data, xarray.Dataset):
                self.zslice = self.data.sel(twt=self.data.twt[0], method="nearest").transpose("iline", "xline").data
                title = "zslice=" + str(self.data.twt[0].data)
            elif isinstance(self.data, np.ndarray) or isinstance(self.data, da.core.Array):
                self.zslice = self.data[:, :, 0]
                title = "zslice=0"
        else:
            if isinstance(self.data, xarray.Dataset):
                self.zslice = self.data.sel(twt=change.new, method="nearest").transpose("iline", "xline").data
            elif isinstance(self.data, np.ndarray) or isinstance(self.data, da.core.Array):
                self.zslice = self.data[:, :, change.new]
            title = "zslice=" + str(change.new)
        self.update(self.zslice, title)
        
    def update_dropdown(self, change):
        if isinstance(self.data, xarray.Dataset):
            self.iline_slider.value = self.data.iline[0]
            self.xline_slider.value = self.data.xline[0]
            self.zslice_slider.value = self.data.twt[0]
        elif isinstance(self.data, np.ndarray) or isinstance(self.data, da.core.Array):
            self.iline_slider.value = 0
            self.xline_slider.value = 0
            self.zslice_slider.value = 0

        if change and change.new == "zslice":
            self.iline_slider.disabled = True
            self.xline_slider.disabled = True
            self.zslice_slider.disabled = False
            self.iline_slider.style.handle_color = 'transparent'
            self.xline_slider.style.handle_color = 'transparent'
            self.zslice_slider.style.handle_color = self.slider_color
            self.update_zslice(None)
        elif change and change.new == "xline":
            self.iline_slider.disabled = True
            self.xline_slider.disabled = False
            self.zslice_slider.disabled = True
            self.iline_slider.style.handle_color = 'transparent'
            self.xline_slider.style.handle_color = self.slider_color
            self.zslice_slider.style.handle_color = 'transparent'
            self.update_xline(None)
        else:
            self.iline_slider.disabled = False
            self.xline_slider.disabled = True
            self.zslice_slider.disabled = True
            self.iline_slider.style.handle_color = self.slider_color
            self.xline_slider.style.handle_color = 'transparent'
            self.zslice_slider.style.handle_color = 'transparent'
            self.update_iline(None)
            
    def update_plot_type(self, change):
        if change.new == "raw":
            self.contour = 0
            self.colors_dropdown.disabled = False
        elif change.new == 'contour':
            self.contour = 1
            self.colors_dropdown.disabled = True
        elif change.new == 'contourf':
            self.contour = 2
            self.colors_dropdown.disabled = True
            
        self.redraw()
        
    def update_colors_dropdown(self, change):
        self.cmap = change.new
        
        self.redraw()

    def redraw(self):
        tslice = self.dropdown.value
        if tslice == "iline":
            if isinstance(self.data, xarray.Dataset):
                self.iline = self.data.sel(iline=self.iline_slider.value).transpose("twt", "xline").data
            elif isinstance(self.data, np.ndarray) or isinstance(self.data, da.core.Array):
                self.iline = self.data[self.iline_slider.value, :, :]
            title = "iline=" + str(self.iline_slider.value)
            self.update(self.iline, title)
        elif tslice == 'xline':
            if isinstance(self.data, xarray.Dataset):
                self.xline = self.data.sel(xline=self.xline_slider.value).transpose("twt", "iline").data
            elif isinstance(self.data, np.ndarray) or isinstance(self.data, da.core.Array):
                self.xline = self.data[:, self.xline_slider.value, :]
            title = "xline=" + str(self.xline_slider.value)
            self.update(self.xline, title)
        else:
            if isinstance(self.data, xarray.Dataset):
                self.zslice = self.data.sel(twt=self.zslice_slider.value, method="nearest").transpose("iline", "xline").data
            elif isinstance(self.data, np.ndarray) or isinstance(self.data, da.core.Array):
                self.zslice = self.data[:, :, self.zslice_slider.value]
            title = "zslice=" + str(self.zslice_slider.value)
            self.update(self.zslice, title)


class PlotSEGYDistributed(Operator):
    def __init__(self):
        super().__init__(name="Plot SEG-Y Distributed")

    def run(self, data):
        return PlotSEGYDistributedInternal(data)


class PlotSEGYPredictDistributedInternal(widgets.VBox):
    def __init__(self, xarray, model):
        super().__init__()
        self.output = widgets.Output()
        
        self.contour = 0
        
        self.xarray = xarray
        self.model = model
 
        with self.output:
            fig = plt.figure(figsize=(11, 8))
            
            self.ax = fig.gca()
        
        self.iidx = self.xarray.iline[0]
        self.xidx = self.xarray.xline[0]
        self.zidx = self.xarray.twt[0]
        
        self.iline = self.xarray.sel(iline=self.iidx).transpose("twt", "xline").data
        self.xline = self.xarray.sel(xline=self.xidx).transpose("twt", "iline").data
        self.zslice = self.xarray.sel(twt=self.zidx, method="nearest").transpose("iline", "xline").data
        
        imshow_kwargs = dict(
            cmap="gray", aspect="auto", interpolation="bicubic"
        )
        self.ax.imshow(self.iline.values, **imshow_kwargs)
 
        # define widgets
        self.iline_slider = widgets.IntSlider(
            value=self.iidx, 
            min=self.xarray.iline[0], 
            max=self.xarray.iline[-1], 
            step=self.xarray.iline[1] - self.xarray.iline[0], 
            description='iline:',
            continuous_update=False
        )
        self.xline_slider = widgets.IntSlider(
            value=self.xidx, 
            min=self.xarray.xline[0], 
            max=self.xarray.xline[-1], 
            step=self.xarray.xline[1] - self.xarray.xline[0], 
            description='xline:',
            continuous_update=False
        )
        self.zslice_slider = widgets.IntSlider(
            value=self.zidx, 
            min=self.xarray.twt[0], 
            max=self.xarray.twt[-1], 
            step=self.xarray.twt[1] - self.xarray.twt[0], 
            description='zslice:',
            continuous_update=False
        )
        
        self.dropdown = widgets.Dropdown(
            value='iline', 
            options=['iline', 'xline', 'zslice'], 
            description='Type:'
        )
        
        self.plot_type = widgets.RadioButtons(
            options=['raw', 'contour', 'contourf'],
            description='Plot type:',
            disabled=False
        )
        
        self.colors_dropdown = widgets.Dropdown(
            value='gray', 
            options=['gray', 'seismic', 'rainbow', 'coolwarm'], 
            description='Plot color:'
        )
        
        self.cmap = 'gray'        
 
        main_controls = widgets.VBox([
            self.iline_slider,
            self.xline_slider,
            self.zslice_slider,
            self.dropdown
        ])
        main_controls.layout = make_box_layout()
        
        plot_controls = widgets.VBox([
            self.plot_type,
            self.colors_dropdown
        ])
        plot_controls.layout = make_box_layout()
        
        controls = widgets.HBox([
            main_controls,
            plot_controls
        ])
         
        out_box = widgets.Box([self.output])
        self.output.layout = make_box_layout()
 
        # observe stuff
        self.iline_slider.observe(self.update_iline, 'value')
        self.xline_slider.observe(self.update_xline, 'value')
        self.zslice_slider.observe(self.update_zslice, 'value')
        self.dropdown.observe(self.update_dropdown, 'value')
        self.plot_type.observe(self.update_plot_type, 'value')
        self.colors_dropdown.observe(self.update_colors_dropdown, 'value')
         
        # add to children
        self.children = [controls, self.output]
        
        self.slider_color = self.iline_slider.style.handle_color
        
        self.update_dropdown(None)
        
        self.update(self.xarray.sel(iline=self.xarray.iline[0]).transpose("twt", "xline").data, "iline=" + str(self.xarray.iline[0].data))
     
    def update(self, default_plot, title):
        """Draw line in plot"""
        self.ax.clear()
        
        imshow_kwargs = dict(
            cmap=self.cmap, aspect="auto", interpolation="bicubic",
        )
        
        self.ax.set_title(title)
        
        
        if self.contour:
            if self.contour == 1:
                default_plot.plot.contour(yincrease=False, add_colorbar=False)
            elif self.contour == 2:
                default_plot.plot.contourf(yincrease=False, add_colorbar=False)
            #self.ax.contour(contour)
        else:
            new_values = self.model.predict(default_plot.values)
            self.ax.imshow(new_values, **imshow_kwargs)

        with self.output:
            self.output.clear_output(wait=True)
 
    def update_iline(self, change):
        if not change:
            self.iline = self.xarray.sel(iline=self.xarray.iline[0]).transpose("twt", "xline").data
            title = "iline=" + str(self.xarray.iline[0].data)
        else:
            self.iline = self.xarray.sel(iline=change.new).transpose("twt", "xline").data
            title = "iline=" + str(change.new)
        self.update(self.iline, title)
 
    def update_xline(self, change):
        if not change:
            self.xline = self.xarray.sel(xline=self.xarray.xline[0]).transpose("twt", "iline").data
            title = "xline=" + str(self.xarray.xline[0].data)
        else:
            self.xline = self.xarray.sel(xline=change.new).transpose("twt", "iline").data
            title = "xline=" + str(change.new)
        self.update(self.xline, title)
 
    def update_zslice(self, change):
        if not change:
            self.zslice = self.xarray.sel(twt=self.xarray.twt[0], method="nearest").transpose("iline", "xline").data
            title = "zslice=" + str(self.xarray.twt[0].data)
        else:
            self.zslice = self.xarray.sel(twt=change.new, method="nearest").transpose("iline", "xline").data
            title = "zslice=" + str(change.new)
        self.update(self.zslice, title)
        
    def update_dropdown(self, change):
        if change and change.new == "zslice":
            self.iline_slider.disabled = True
            self.xline_slider.disabled = True
            self.zslice_slider.disabled = False
            self.iline_slider.style.handle_color = 'transparent'
            self.xline_slider.style.handle_color = 'transparent'
            self.zslice_slider.style.handle_color = self.slider_color
            self.iline_slider.value = self.xarray.iline[0]
            self.xline_slider.value = self.xarray.xline[0]
            self.zslice_slider.value = self.xarray.twt[0]
            self.update_zslice(None)
        elif change and change.new == "xline":
            self.iline_slider.disabled = True
            self.xline_slider.disabled = False
            self.zslice_slider.disabled = True
            self.iline_slider.style.handle_color = 'transparent'
            self.xline_slider.style.handle_color = self.slider_color
            self.zslice_slider.style.handle_color = 'transparent'
            self.iline_slider.value = self.xarray.iline[0]
            self.xline_slider.value = self.xarray.xline[0]
            self.zslice_slider.value = self.xarray.twt[0]
            self.update_xline(None)
        else:
            self.iline_slider.disabled = False
            self.xline_slider.disabled = True
            self.zslice_slider.disabled = True
            self.iline_slider.style.handle_color = self.slider_color
            self.xline_slider.style.handle_color = 'transparent'
            self.zslice_slider.style.handle_color = 'transparent'
            self.iline_slider.value = self.xarray.iline[0]
            self.xline_slider.value = self.xarray.xline[0]
            self.zslice_slider.value = self.xarray.twt[0]
            self.update_iline(None)
            
    def update_plot_type(self, change):
        if change.new == "raw":
            self.contour = 0
            self.colors_dropdown.disabled = False
        elif change.new == 'contour':
            self.contour = 1
            self.colors_dropdown.disabled = True
        elif change.new == 'contourf':
            self.contour = 2
            self.colors_dropdown.disabled = True
            
        self.redraw()
        
    def update_colors_dropdown(self, change):
        self.cmap = change.new
        
        self.redraw()

    def redraw(self):
        tslice = self.dropdown.value
        if tslice == "iline":
            self.iline = self.xarray.sel(iline=self.iline_slider.value).transpose("twt", "xline").data
            title = "iline=" + str(self.iline_slider.value)
            self.update(self.iline, title)
        elif tslice == 'xline':
            self.xline = self.xarray.sel(xline=self.xline_slider.value).transpose("twt", "iline").data
            title = "xline=" + str(self.xline_slider.value)
            self.update(self.xline, title)
        else:
            self.zslice = self.xarray.sel(twt=self.zslice_slider.value, method="nearest").transpose("iline", "xline").data
            title = "zslice=" + str(self.zslice_slider.value)
            self.update(self.zslice, title)


class PlotSEGYPredictDistributed(Operator):
    def __init__(self):
        self.name = "Plot SEG-Y Predict Distributed"
        super().__init__(name=self.name)

    def plot_results(self, results):
        ids = list(results.result.keys())

        task_ref = None
        for item in ids:
            if item.name == self.name:
                task_ref = item
                break

        last = results.result[task_ref]

        return last._result.value

    def run(self, data, model):
        return PlotSEGYPredictDistributedInternal(data, model)
