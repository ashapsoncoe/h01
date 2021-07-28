import neuroglancer
import json
import numpy as np
import time


em = 'brainmaps://964355253395:h01:goog14r0_8nm'
agglo = 'brainmaps://964355253395:h01:goog14r0seg1_agg20200916c3_flat'
input_data_dir = 'c:/work/final/excitatory_connections_in_layer_1.json'
voxel_size = [8,8,33]


class ConnectionChecker():

    def __init__(self, input_data_dir, em, agglo, voxel_size):

        self.viewer = neuroglancer.Viewer()
        self.voxel_size = voxel_size

        self.change_location([0,0,0])

        with self.viewer.txn() as s:

            s.layers['em'] = neuroglancer.ImageLayer(source = em)
            s.layers['agglomeration'] = neuroglancer.SegmentationLayer(source = agglo, segment_colors={})
            s.layers['synapses'] = neuroglancer.AnnotationLayer()

        self.viewer.actions.add('true_connection', lambda s: self.mark_as_tf('true'))
        self.viewer.actions.add('false_connection', lambda s: self.mark_as_tf('false'))
        self.viewer.actions.add('go_back', lambda s: self.changepos(-1))
        self.viewer.actions.add('go_forward', lambda s: self.changepos(+1))

        with self.viewer.config_state.txn() as s:

            s.input_event_bindings.viewer['keyt'] = 'true_connection'
            s.input_event_bindings.viewer['keyf'] = 'false_connection'
            s.input_event_bindings.viewer['keyk'] = 'go_forward'
            s.input_event_bindings.viewer['keyj'] = 'go_back'


        with open(input_data_dir, 'r') as fp:
            self.input_data = json.load(fp)

        for pos, datum in enumerate(self.input_data):

            self.pos = pos

            if 'connection_decision' not in datum:
                self.update_pair()
                break

        


    def changepos(self, change):

        self.pos += change

        self.update_pair()


    def mark_as_tf(self, decision):

        self.input_data[self.pos]['connection_decision'] = decision

        with open(input_data_dir, 'w') as fp:
            json.dump(self.input_data, fp)

        while 'connection_decision' in self.input_data[self.pos]:
            self.pos += 1
        
        self.update_pair()


    def change_location(self, location):

        with self.viewer.txn() as s:

            dimensions = neuroglancer.CoordinateSpace(scales=self.voxel_size, units='nm', names=['x', 'y', 'z'])
            s.showSlices = False
            s.dimensions = dimensions
            s.crossSectionScale = 1.0
            s.projectionScale = 80000
            s.position = np.array(location)
            

    def update_pair(self):

        pre_seg = self.input_data[self.pos]['pre_seg']
        post_seg = self.input_data[self.pos]['post_seg']
        syn_id = self.input_data[self.pos]['syn_id']
        syn_loc = self.input_data[self.pos]['syn_loc']

        with self.viewer.txn() as s:

            s.layers['agglomeration'].segments = set([pre_seg, post_seg])
            s.layers['agglomeration'].segment_colors[int(pre_seg)] = 'lightblue'
            s.layers['agglomeration'].segment_colors[int(post_seg)] = 'yellow'

            pa = neuroglancer.PointAnnotation(id=syn_id, description=syn_id, point=syn_loc)

            s.layers['synapses'].annotations = [pa]
            s.layers['synapses'].annotationColor = 'white'

        self.change_location(syn_loc)

        msg = f"Position {self.pos} of {len(self.input_data)-1}, type: {self.input_data[self.pos]['connection_type']}"

        if 'connection_decision' in self.input_data[self.pos]:
            prev_decision = self.input_data[self.pos]['connection_decision']
            msg = msg + f' Previous decision: {prev_decision}'

        with self.viewer.config_state.txn() as s:
            s.status_messages['status'] = msg


if __name__ == "__main__":

    c = ConnectionChecker(input_data_dir, em, agglo, voxel_size)

    print(c.viewer)

    time.sleep(1000000)




