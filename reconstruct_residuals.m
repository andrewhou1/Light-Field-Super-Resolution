function [] = reconstruct_residuals(input_dir, output_dir)
horizontal_EPIs = dir(fullfile(input_dir, '*.png'));
original_EPIs = dir(fullfile('upsampled_EPIs2/', '*.png'));

if ~exist( output_dir, 'dir' )
    mkdir( output_dir )
end

% Cycle through rows
for i = 1:9
    imgs = zeros(800, 1400, 8);
    start = (i-1)*800+1;
    finish = start+799;
    for j = start:finish
        % Output EPI - YCbCr, just the Y channel
        EPI = imread( [input_dir horizontal_EPIs(j).name] );
        EPI = double(EPI);
        % remove compressing by YCbCr conversion
        EPI_n = ((EPI / 255) * 219) + 16;
        
        % Input upsampled EPI - RGB space
        orig_EPI_rgb = imread( ['upsampled_EPIs2' filesep original_EPIs(j).name] );
        orig_EPI_rgb = double(orig_EPI_rgb);
        orig_EPI_ycbcr = rgb2ycbcr( orig_EPI_rgb );
        orig_EPI = orig_EPI_ycbcr(:,:,1);
        %orig_EPI = ((orig_EPI / 255) * 219) + 16;
        
        % In range 0 to 1, where 0.5 is no residual value (no difference)
        residual = (((EPI_n - orig_EPI) / 255) + 1) / 2;
        
        for k = 1:8
            imgs(j-start+1, :, k) = residual(2*k, :, :);
        end
    end
    
    for f = 1:8
        % Output range
        im = imgs(:, :, f);
        fno = sprintf( '%02d', f );
        ino = sprintf( '%02d', 2*(i-1)+1 );
        imwrite(im, [output_dir ino '-' fno '.png'] );
    end
end
end