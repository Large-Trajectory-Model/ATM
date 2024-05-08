import numpy as np

from atm.dataloader.base_dataset import BaseDataset
from atm.utils.flow_utils import sample_tracks_visible_first


class ATMPretrainDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self._index_to_view_id = {}
        super().__init__(*args, **kwargs)

    def load_demo_info(self):
        start_idx = 0
        for demo_idx, fn in enumerate(self.buffer_fns):
            demo = self.load_h5(fn)

            if self.views is None:
                self.views = list(demo["root"].keys())
                self.views.remove("actions")
                self.views.remove("task_emb_bert")
                self.views.remove("extra_states")
                self.views.sort()

            demo_len = demo["root"][self.views[0]]["video"][0].shape[0]

            if self.cache_all:
                demo = self.process_demo(demo)
                for v in self.views:
                    del demo["root"][v]["video"]
                self._cache.append(demo)
            self._demo_id_to_path[demo_idx] = fn
            self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len*2)})
            self._index_to_view_id.update({k: (k - start_idx) % 2 for k in range(start_idx, start_idx + demo_len*2)})
            self._demo_id_to_start_indices[demo_idx] = start_idx
            self._demo_id_to_demo_length[demo_idx] = demo_len
            start_idx += demo_len * 2

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx

    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        view = self.views[self._index_to_view_id[index]]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = (index - demo_start_index) // 2

        if self.cache_all:
            demo = self._cache[demo_id]
            if self.cache_image:
                vids = self._load_image_list_from_demo(demo, view, time_offset, backward=True)  # t c h w
            else:
                vids = self._load_image_list_from_disk(demo_id, view, time_offset, backward=True)  # t c h w
        else:
            demo_pth = self._demo_id_to_path[demo_id]
            demo = self.process_demo(self.load_h5(demo_pth))
            vids = self._load_image_list_from_demo(demo, view, time_offset, backward=True)  # t c h w

        tracks = demo["root"][view]["tracks"][time_offset:time_offset + self.num_track_ts]  # track_len n 2
        vis = demo["root"][view]['vis'][time_offset:time_offset + self.num_track_ts]  # track_len n
        task_emb = demo["root"]["task_emb_bert"]  # (dim,)

        # augment videos
        if np.random.rand() < self.aug_prob:
            vids = vids[None]  # expand to (1, t, c, h, w) to fit the input shape of random shift augmentation
            tracks = tracks[None, None]  # expand to (1, 1, track_len, n, 2) to fit the input shape of random shift augmentation
            vids, tracks = self.augmentor((vids / 255., tracks))
            vids = vids[0, ...] * 255.
            tracks = tracks[0, 0, ...]

        # sample tracks
        tracks, vis = sample_tracks_visible_first(tracks, vis, num_samples=self.num_track_ids)

        return vids, tracks, vis, task_emb
