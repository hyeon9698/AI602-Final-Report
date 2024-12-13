#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy  # Added to avoid issue
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import open3d as o3d
import numpy as np

def l1_loss_mask(network_output, gt, mask = None):
    if mask is None:
        return l1_loss(network_output, gt)
    else:
        return torch.abs((network_output - gt) * mask).sum() / mask.sum()

def loss_photometric(image, gt_image, opt, valid=None):
    Ll1 =  l1_loss_mask(image, gt_image, mask=valid)
    loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))
    return loss

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    N_gaussian_models = 2
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    ####
    #### 코드 수정
    ####
    GsDict = {}
    for i in range(N_gaussian_models):
        if i == 0:
            GsDict[f"gs{i}"] = gaussians
        elif i > 0:
            GsDict[f"gs{i}"] = GaussianModel(dataset.sh_degree)
            GsDict[f"gs{i}"].create_from_pcd(scene.init_point_cloud, scene.cameras_extent)
            GsDict[f"gs{i}"].training_setup(opt)
            print(f"Create gaussians{i}")
    print(f"GsDict.keys() is {GsDict.keys()}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            for i in range(N_gaussian_models):
                GsDict[f"gs{i}"].oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        RenderDict = {}
        LossDict = {}

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        for i in range(N_gaussian_models):
            RenderDict[f"render_pkg_gs{i}"] = render(viewpoint_cam, GsDict[f'gs{i}'], pipe, bg)
            RenderDict[f"image_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["render"]
            # RenderDict[f"depth_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["depth"]
            # RenderDict[f"alpha_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["alpha"]
            RenderDict[f"viewspace_point_tensor_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["viewspace_points"]
            RenderDict[f"visibility_filter_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["visibility_filter"]
            RenderDict[f"radii_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["radii"]
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()

        # Loss
        for i in range(N_gaussian_models):
            LossDict[f"loss_gs{i}"] = loss_photometric(RenderDict[f"image_gs{i}"], gt_image, opt=opt)
        loss = LossDict["loss_gs0"]
        for i in range(N_gaussian_models):
            LossDict[f"loss_gs{i}"].backward()

        # Ll1 = l1_loss(image, gt_image)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                for i in range(N_gaussian_models):
                    viewspace_point_tensor = RenderDict[f"viewspace_point_tensor_gs{i}"]
                    visibility_filter = RenderDict[f"visibility_filter_gs{i}"]
                    radii = RenderDict[f"radii_gs{i}"]
                    GsDict[f"gs{i}"].max_radii2D[visibility_filter] = torch.max(GsDict[f"gs{i}"].max_radii2D[visibility_filter], radii[visibility_filter])
                    GsDict[f"gs{i}"].add_densification_stats(viewspace_point_tensor, visibility_filter)

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()
                # density and prune
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    # size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    for i in range(N_gaussian_models):

                        GsDict[f"gs{i}"].densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)                              


            # Optimizer step
            # if iteration < opt.iterations:
            #     gaussians.optimizer.step()
            #     gaussians.optimizer.zero_grad(set_to_none = True)
            if iteration < opt.iterations:
                for i in range(N_gaussian_models):
                    GsDict[f"gs{i}"].optimizer.step()
                    GsDict[f"gs{i}"].optimizer.zero_grad(set_to_none = True)
            for i in range(N_gaussian_models):
                GsDict[f"gs{i}"].update_learning_rate(iteration)
                if (iteration - 2000 - 1) % opt.opacity_reset_interval == 0 and \
                        iteration > 2000:
                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print(f"reset opacity of gaussians-{i} at iteration {iteration}")
                    GsDict[f"gs{i}"].reset_opacity()
            if N_gaussian_models > 1 and iteration > opt.densify_from_iter and iteration % 500 == 0:
                for i in range(N_gaussian_models):
                    for j in range(N_gaussian_models):
                        if i != j:
                            source_cloud = o3d.geometry.PointCloud()
                            source_cloud.points = o3d.utility.Vector3dVector(GsDict[f"gs{i}"].get_xyz.clone().cpu().numpy())
                            target_cloud = o3d.geometry.PointCloud()
                            target_cloud.points = o3d.utility.Vector3dVector(GsDict[f"gs{j}"].get_xyz.clone().cpu().numpy())
                            trans_matrix = np.identity(4)
                            threshold = 5
                            evaluation = o3d.pipelines.registration.evaluate_registration(source_cloud, target_cloud, threshold, trans_matrix)
                            correspondence = np.array(evaluation.correspondence_set)
                            mask_consistent = torch.zeros((GsDict[f"gs{i}"].get_xyz.shape[0], 1)).cuda()
                            mask_consistent[correspondence[:, 0], :] = 1
                            GsDict[f"indice_consistent_gs{i}to{j}"] = correspondence
                            GsDict[f"mask_inconsistent_gs{i}"] = ~(mask_consistent.bool())
                for i in range(N_gaussian_models):
                    GsDict[f"gs{i}"].prune_from_mask(GsDict[f"mask_inconsistent_gs{i}"].squeeze(), iter=iteration)
            # if (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
