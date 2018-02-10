function [] = residuals_groundtruth(generated_dir, groundtruth_dir, output_dir)
groundtruth_views = dir(fullfile(groundtruth_dir, '*.png'));
generated_views = dir(fullfile(generated_dir, '*.png'));

if ~exist( output_dir, 'dir' )
    mkdir( output_dir )
end

for i = 1:9
    for j = 1:8
        gtFile = [groundtruth_dir groundtruth_views(2*(i-1)*17+2*j).name];
        groundtruth = rgb2ycbcr( imread( gtFile ) );
        % Between 0 and 255
        ychannel = double(groundtruth(:, :, 1));
        
        genFile = [generated_dir, generated_views((i-1)*8+j).name];
        % Between 0 and 255
        generated_view = double( imread( genFile ) );
        
        % I think that generated view has been normalized on output
        % That values between 16 and 235 were clamped to 0 to 255
        % So, let's try...
        generated_view = ((generated_view / 255) * 219) + 16;
        
        % Place residual between 0 and 1 with no difference (0) at 0.5
        % gen_view - ychannel is -255 to 255
        % then to -1 to 1
        % then to 0 to 2
        % then to 0 to 1, with 0 at 0.5
        residual = (((generated_view - ychannel) / 255) + 1) / 2;
        
        % Write output
        imwrite( residual, [output_dir, '0', num2str(2*(i-1)+1), '-', '0', num2str(2*j), '.png']);
    end
end
end