import tensorflow.compat.v1 as tf
import tensorflow as tf2
tf.disable_eager_execution()

def convert_posenet_model(sess):
    print('Converting model...')
    export_dir = './converted_models/saved_model'
    input_tensor_name = 'image:0'  # just an example
    offsets = 'offset_2:0'
    displacement_fwd = 'displacement_fwd_2:0'
    displacement_bwd = 'displacement_bwd_2:0'
    heatmaps = 'heatmap:0'
    export_tf1(sess, input_tensor_name, [offsets, displacement_fwd, displacement_bwd, heatmaps], sess, export_dir)
    tf.saved_model.save(Export(), export_dir)
    print('New model saved to ' + export_dir)


def export_tf1(session, in_tnsr_fullname, out_tnsrS_fullname, sess, export_dir='./export'):
    assert isinstance(in_tnsr_fullname, str)
    assert all([isinstance(out_tnsr_fullname, str) for out_tnsr_fullname in out_tnsrS_fullname])
    in_tnsr_name = in_tnsr_fullname.split(':')[0]
    out_tnsrS_name = [out_tnsr_fullname.split(':')[0] for out_tnsr_fullname in out_tnsrS_fullname]

    graph_def = tf.graph_util.convert_variables_to_constants(session, session.graph.as_graph_def(), out_tnsrS_name)

    # tf.reset_default_graph()
    outs = tf.import_graph_def(graph_def, name="", return_elements=out_tnsrS_fullname)
    g = outs[0].graph

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    input_signatures = {in_tnsr_name: g.get_tensor_by_name(in_tnsr_fullname)}
    output_signatures = {}
    for out_tnsr_name, out_tnsr_fullname in zip(out_tnsrS_name, out_tnsrS_fullname):
        output_signatures[out_tnsr_name] = g.get_tensor_by_name(out_tnsr_fullname)
    signature = tf.saved_model.signature_def_utils.predict_signature_def(input_signatures, output_signatures)

    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature},
    )

    builder.save()

class Export(tf.Module):
    def __init__(self):
        super(Export, self).__init__()
        tf1_saved_model_directory = 'converted_models/saved_model'  # just an example
        self.tf1_model = tf2.saved_model.load(tf1_saved_model_directory)
        input_tensor_name = 'image:0'  # just an example
        offsets = 'offset_2:0'
        displacement_fwd = 'displacement_fwd_2:0'
        displacement_bwd = 'displacement_bwd_2:0'
        heatmaps = 'heatmap:0'
        self.tf1_model = self.tf1_model.prune(input_tensor_name, [offsets, displacement_fwd, displacement_bwd, heatmaps])

    @tf2.function
    def __call__(self, x):
        out = self.tf1_model(x)
        return out