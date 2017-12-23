function [PSNRs] = computePSNR(generated_dir, groundtruth_dir)
PSNRs = zeros(9, 8);
groundtruth_views = dir(fullfile(groundtruth_dir, '*.png'));
generated_views = dir(fullfile(generated_dir, '*.png'));

for i = 1:9
    for j = 1:8
        groundtruth = rgb2ycbcr(imread(strcat(groundtruth_dir, groundtruth_views(2*(i-1)*17+2*j).name)));
        ychannel = groundtruth(:, :, 1);
        generated_view = imread(strcat(generated_dir, generated_views((i-1)*8+j).name));
        generated_view = generated_view(:, :, 1);
        [peaksnr, snr] = psnr(generated_view, ychannel, 255);
        PSNRs(i, j) = peaksnr;
    end
end

end