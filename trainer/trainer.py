import gc
import time
import torch
import numpy as np
from tqdm import tqdm
from config.all_config import gen_log
from config.base_config import Config
from collections import defaultdict, deque
from trainer.base_trainer import BaseTrainer
from modules.metrics import sim_matrix_training, np_softmax, generate_embeds_per_video_id, sim_matrix_inference, \
    sim_matrix_inference_light_allops, sim_matrix_training_text


class Trainer(BaseTrainer):

    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data_loader,
                 test_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best = -1.0

        self.test_batch_size = config.batch_size

    def _train_epoch(self, epoch):

        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        eval_steps = np.linspace(0, num_steps - 1, self.evals_per_epoch + 1, dtype=int)[1:]

        start_time = time.time()

        for batch_idx, data in enumerate(self.train_data_loader):
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

            data['video'] = data['video'].to(self.device)

            text_features, text_pooled, video_features, video_pooled = self.model(data, is_train=True)

            output = sim_matrix_training(text_features, video_pooled, self.pooling_type)
            loss_otv = self.loss(output, self.model.clip.logit_scale)
            output = sim_matrix_training(text_pooled, video_pooled, self.pooling_type)
            loss_etv = self.loss(output, self.model.clip.logit_scale)
            loss_tv = loss_otv + loss_etv

            output = sim_matrix_training_text(text_features, text_pooled, self.pooling_type)
            loss_tt = self.loss(output, self.model.clip.logit_scale)

            loss_all = loss_tv + loss_tt

            loss_all.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1

            total_loss += loss_all.detach().item()

            cost_time = time.time() - start_time
            start_time = time.time()

            eta_time = (len(self.train_data_loader) * self.config.num_epochs - batch_idx * epoch) * cost_time
            eta_time = f"{int(eta_time // 3600):02}:{int((eta_time % 3600) // 60):02}:{int(eta_time % 60):02}"

            if batch_idx % self.log_step == 0:
                msg = (
                'Train epoch: {} dl:{}/{} total_loss:{:.10f}, eta_time:{}'.format(
                    epoch,
                    batch_idx,
                    num_steps - 1,
                    loss_all.detach().item(),
                    eta_time
                ))
                gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)

            if batch_idx in eval_steps or (epoch == 1 and batch_idx == 0):

                if self.config.skip_eval:
                    msg = '\nSkip eval due to long time usage!\n'
                    gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)

                else:
                    test_res, Rsum = self._valid_epoch_step(epoch, batch_idx, num_steps - 1)
                    self.model.train()

                    if Rsum > self.best:
                        self.best = Rsum
                        self._save_checkpoint(epoch, save_best=True)

                    msg = (" Current Best Text-Video R@sum is {}".format(self.best))
                    gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)
                    gen_log(model_path=self.config.model_path, log_name='log_test', msg=msg)

        res = {
            'loss_train': total_loss / num_steps
        }

        return res

    def _valid_epoch_step(self, epoch, step, num_steps):

        self.model.eval()
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []

        start_selection_time = time.time()

        with torch.no_grad():
            for idx, data in tqdm(enumerate(self.test_data_loader)):
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                data['video'] = data['video'].to(self.device)

                text_features, video_features = self.model(data, is_train=False)
                text_embed_arr.append(text_features.cpu())
                vid_embed_arr.append(video_features.cpu())
                for v_id in data['video_id']:
                    all_vid_ids.append(v_id)

            text_embeds = torch.cat(text_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)

            vid_embeds_per_video_id = {}
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id:
                    vid_embeds_per_video_id[v_id] = vid_embeds[idx]
            vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])  # [N, F, D]

            self.model.video_transformer.cpu()
            # num_vids x num_texts x embed_dim
            vid_embeds_pooled = self.model.video_transformer(text_embeds, vid_embeds)
            self.model.video_transformer.cuda()

            self.model.text_transformer.cpu()
            # num_texts x num_vids x embed_dim
            text_embeds_pooled = self.model.text_transformer(text_embeds, vid_embeds)
            self.model.text_transformer.cuda()

            text_embeds_choice_allpairs = torch.zeros(size=(vid_embeds.shape[0], text_embeds.shape[0], text_embeds.shape[1]))
            text_embeds_pooled = text_embeds_pooled.permute(1, 0, 2)
            for (idx, single_text_embed_pooled), single_vid_embed_pooled in tqdm(zip(enumerate(text_embeds_pooled), vid_embeds_pooled)):

                all_text_embed_choice = []

                # We add other Enhance-Text-Embed!
                all_text_embed_choice.append(single_text_embed_pooled)
                all_text_embed_choice.append(text_embeds)
                all_text_embed_choice_arr = torch.stack(all_text_embed_choice, dim=0)      # [M, N, D]

                all_text_embed_choice_arr = all_text_embed_choice_arr / all_text_embed_choice_arr.norm(dim=-1, keepdim=True)
                single_vid_embed_pooled = single_vid_embed_pooled / single_vid_embed_pooled.norm(dim=-1, keepdim=True)

                # [M, N] <<== [M, N, D] @ [N, D]
                sim_select = torch.sum(torch.mul(all_text_embed_choice_arr, single_vid_embed_pooled), dim=-1)

                # [B]
                max_indices = torch.argmax(sim_select, dim=0)

                selected_plane = torch.ones((all_text_embed_choice_arr.shape[1], all_text_embed_choice_arr.shape[2]))
                for i in range(all_text_embed_choice_arr.shape[1]):
                    selected_plane[i, :] = all_text_embed_choice_arr[max_indices[i], i, :]
                text_embeds_choice_allpairs[idx, :, :] = selected_plane

            del text_embeds, vid_embeds
            gc.collect()

            text_embeds_per_video_id, vid_embeds_pooled_per_video_id = generate_embeds_per_video_id(text_embeds_choice_allpairs, vid_embeds_pooled, all_vid_ids, self.pooling_type)

            if self.config.save_memory_mode:
                sims = sim_matrix_inference_light_allops(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type, self.config.batch_size_split, self.config)
            else:
                sims = sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, self.pooling_type)

            del text_embeds_per_video_id, vid_embeds_pooled_per_video_id
            gc.collect()

            if self.config.DSL:
                sims_t2v = sims * np_softmax(sims * 100, axis=0)
            else:
                sims_t2v = sims
            metrics = self.metrics
            res = metrics(sims_t2v)
            Rsum = res['Rsum']
            msg = (f"--text-video--Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                   f"R@1: {res['R1']:.1f}",
                   f"R@5: {res['R5']:.1f}",
                   f"R@10: {res['R10']:.1f} ",
                   f"R@sum: {res['Rsum']:.1f} ",
                   f"MedR: {res['MdR']:.1f}",
                   f"MnR: {res['MnR']:.1f}",
                   )
            gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)
            gen_log(model_path=self.config.model_path, log_name='log_test', msg=msg)

            ''' Here we conduct video-text retrieval (.T)'''
            sims = sims.permute(2, 1, 0)
            if self.config.DSL:
                sims_v2t = sims * np_softmax(sims * 100, axis=0)
            else:
                sims_v2t = sims
            res = metrics(sims_v2t)
            msg = (f"--video-text--Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                   f"R@1: {res['R1']:.1f}",
                   f"R@5: {res['R5']:.1f}",
                   f"R@10: {res['R10']:.1f} ",
                   f"Rsum: {res['Rsum']:.1f} ",
                   f"MedR: {res['MdR']:.1f}",
                   f"MnR: {res['MnR']:.1f}",
                   )
            gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)
            gen_log(model_path=self.config.model_path, log_name='log_test', msg=msg)

            end_selection_time = time.time()

            msg = (
                f'To compute all video-text embeddings for the whole dataset, the time usage is {end_selection_time - start_selection_time}')
            gen_log(model_path=self.config.model_path, log_name='log_train', msg=msg)
            gen_log(model_path=self.config.model_path, log_name='log_test', msg=msg)

            return res, Rsum
