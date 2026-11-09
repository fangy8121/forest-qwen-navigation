"""Single observation-to-input implementation shared by conversion and flight."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .multilevel_bev import extract_multilevel_bev, render_multilevel_bev, input_prompt
from .runs import file_hash, atomic_json
from .geometry import center_clearance

DEFAULT_REPRESENTATION = 'multilevel_xyz'


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def render_bev(points, evidence, goal, output, corridor, max_acceleration, clearance=None):
    from .observation import _render_lock
    bev = extract_multilevel_bev(points, evidence, corridor=corridor,
                     max_acceleration=max_acceleration, actual_start=np.zeros(3), center_clearance=clearance)
    with _render_lock:
        render_multilevel_bev(bev, goal, output)
    return bev


def make_input(obs, folder, grid=None):
    folder = Path(folder)
    started = time.perf_counter()
    corridor = obs.get('corridor')
    if corridor is not None and not len(corridor):
        corridor = None
    flight = json.loads(str(obs['flight_json']))
    representation = flight.get('input_representation', DEFAULT_REPRESENTATION)
    if representation not in (DEFAULT_REPRESENTATION, 'volume_atlas_v1'):
        raise ValueError('unknown waypoint input representation: '+str(representation))
    image = folder/'input.png'
    if representation == 'volume_atlas_v1':
        from .volume_bev import NAME, HEIGHTS, SIZE, grid_states, render_atlas, input_prompt as volume_prompt
        states = grid_states(obs, grid)
        render_atlas(states, image)
        prompt = volume_prompt(obs)
        volume = folder/'input_volume.npz'
        np.savez_compressed(volume, states=states, heights=HEIGHTS)
        atomic_json(folder/'prompt.json', dict(prompt=prompt))
        return dict(image=str(image),image_sha256=file_hash(image),prompt=prompt,prompt_sha256=text_hash(prompt),
                    prepare_seconds=time.perf_counter()-started,heights_m=HEIGHTS.tolist(),allow_vertical=True,
                    actual_body_support=True,safety_profile=flight['safety_profile'],representation=NAME,
                    image_size=list(SIZE),volume=str(volume),volume_sha256=file_hash(volume))
    bev = render_bev(obs['points'], obs['evidence'], obs['goal'], image,
                     corridor, flight['max_acceleration'], center_clearance(flight))
    prompt = input_prompt(bev, obs['goal'], obs['velocity'], obs['acceleration'],
                          allow_vertical=True, corridor=corridor,
                          verified_stop_only=center_clearance(flight) is not None,
                          goal_horizon=obs.get('goal_horizon'))
    atomic_json(folder/'prompt.json', dict(prompt=prompt))
    return dict(image=str(image), image_sha256=file_hash(image), prompt=prompt,
                prompt_sha256=text_hash(prompt), prepare_seconds=time.perf_counter()-started,
                representation=representation,
                heights_m=bev.heights.tolist(), allow_vertical=True, actual_body_support=True,
                safety_profile=flight.get('safety_profile', 'cylinder_xyz_v1'))
