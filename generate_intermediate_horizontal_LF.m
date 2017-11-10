horizontal_EPIs = dir(fullfile('upsampled_EPIs2/', '*.png'));

for i = 1:9
    imgs = zeros(800, 1400, 8);
    start = (i-1)*800+1;
    finish = start+799;
    for j = start:finish
        EPI = imread(strcat('upsampled_EPIs2/', horizontal_EPIs(j).name));
        EPI = rgb2ycbcr(EPI);
        EPI = EPI(:, :, 1);
        for k = 1:8
            imgs(j-start+1, :, k) = EPI(2*k, :, :);
        end
    end
    
    for f = 1:8
        if(2*f < 10)
            if(i < 6)
                imwrite(uint8(imgs(:, :, f)), strcat('LF_noCNN/', '0', num2str(2*(i-1)+1), '-', '0', num2str(2*f), '.png'));
            else
                imwrite(uint8(imgs(:, :, f)), strcat('LF_noCNN/', num2str(2*(i-1)+1), '-', '0', num2str(2*f), '.png'));
            end
        else
            if(i < 6)
                imwrite(uint8(imgs(:, :, f)), strcat('LF_noCNN/', '0', num2str(2*(i-1)+1), '-', num2str(2*f), '.png'));
            else
                imwrite(uint8(imgs(:, :, f)), strcat('LF_noCNN/', num2str(2*(i-1)+1), '-', num2str(2*f), '.png'));
            end
        end
    end
end