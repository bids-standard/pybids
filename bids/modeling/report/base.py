import os
import datetime
from pathlib import Path
from bids.layout import BIDSLayout
from bids.modeling import BIDSStatsModelsGraph
from bids import __version__ as bids_version
import pkg_resources as pkgr
from .utils import deroot, snake_to_camel, displayify, to_alphanum, generate_contrast_matrix

PATH_PATTERNS = [
    'reports/[sub-{subject}/][ses-{session}/][level-{level}_][sub-{subject}_][ses-{session}_]'
    '[run-{run}_]model-{model}.html'
]

def _build_node_dict(node, all_entities):
    report_node = {
        'name': node.name, 
        'analyses': [],
        'group_by': node.group_by,
        'model': node.model,
        'level': node.level,
        'transformations': node.transformations,
        }
    for out in node.outputs_:
        analysis_dict = {'entities': {}, 'contrasts': []}

        for contrast_info in out.contrasts:
            cents = contrast_info.entities.copy()
            cents["level"] = out.node.level
            cents["name"] = out.node.name

            for key in ('datatype', 'desc', 'suffix', 'extension'):
                cents.pop(key, None)
            for key in all_entities:
                cents.pop(key, None)

            for k, v in cents.items():
                if k in ("name", "contrast"):
                    cents.update({k: to_alphanum(str(v))})

            analysis_dict['entities'] = {
                key: val
                for key, val in cents.items()
                if key in ('subject', 'session', 'task', 'run') and val
            }

            analysis_dict['contrasts'].append(
                {
                    'name': displayify(contrast_info.name),
                }
            )

        ents = out.entities.copy()
        # Space doesn't apply to design/contrast matrices, or resolution
        for k in ['space', 'res']:
            ents.pop(k, None)

        analysis_dict['X'] = out.X

        # If reports were run-level
        if out.report_ is not None:
            analysis_dict['design_matrix_plot'] = out.report_['design_matrix_plot'].to_json()
            analysis_dict['design_matrix_corrplot'] = out.report_['design_matrix_corrplot'].to_json()
            analysis_dict['VIF'] = out.report_['VIF']
            analysis_dict['trans_hist'] = out.trans_hist
            
        analysis_dict['contrast_matrix'] = generate_contrast_matrix(out.contrasts, out.X.columns)
        report_node['analyses'].append(analysis_dict)

    return report_node


def _build_report_dict(graph):
    report = {
        'dataset': {
            'name': graph.layout.description['Name'],
        },
        'model': graph.model,
        'nodes': [],
        'version': bids_version,
        'timestamp': datetime.datetime.now(),
        'graph_plot': graph.write_graph(format='svg', pipe=True)
    }

    if 'DatasetDOI' in graph.layout.description:
        report['dataset']['doi'] = graph.layout.description['DatasetDOI']

    all_entities = graph.layout.get_entities(metadata=True)
    for name, node in graph.nodes.items():
        report['nodes'].append(_build_node_dict(node, all_entities))

    # Get subjects hackily
    report['subjects'] = sorted(
        {analysis_dict['entities']['subject'] for analysis_dict in report['nodes'][0]['analyses']}
    )

    return report


def _write_report(report_dict, out_dir, template_path=None):
    try:
        import jinja2
    except ImportError:
        raise ImportError(
            "Jinja2 must be installed to generate reports. "
            "You can install it with pip install jinja2."
        )   

    if template_path is None:
        searchpath = pkgr.resource_filename('bids', '/')
        template_file = 'modeling/report/report_template.jinja'
    else:
        searchpath, template_file = os.path.split(template_path)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath=searchpath))
    tpl = env.get_template(template_file)

    model = snake_to_camel(report_dict['model']['name'])
    target_file = os.path.join(
        out_dir, f"{model}_report.html"
    )

    report_dict = deroot(report_dict, os.path.dirname(target_file))

    html = tpl.render(report_dict)
    Path(target_file).parent.mkdir(parents=True, exist_ok=True)
    Path(target_file).write_text(html)


def generate_report(
    model, dataset_path, derivatives, output_dir, scan_length=None,
    **entities):
    """
    Generate a report for a model.

    Parameters
    ----------
    model : str
        Path to BIDS-StatsModels model.
    dataset_path : str
        Path to BIDS dataset.
    derivatives : str or list
        Path to BIDS derivatives.
    output_dir : str
        Path to output directory.
    scan_length : float (optional)
        Length of scan in seconds. If not provided, will be read from
        dataset (requires image files to be present).
    entities : dict (optional)
        A dictionary of BIDS entities to filter the layout on.
    """
    # Initialize BIDSLayout
    layout = BIDSLayout(dataset_path, derivatives=derivatives)

    # Build BIDSStatsModelsGraph instance
    graph = BIDSStatsModelsGraph(layout, model)

    graph.load_collections(scan_length=scan_length, **entities)

    # Run entire graph
    graph.run_graph(
        scan_length=scan_length, entities=entities, transformation_history=True, 
        node_reports=True)

    report_dict =  _build_report_dict(graph)

    _write_report(report_dict, output_dir)