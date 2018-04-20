function RecursiveBenchmark()
addpath('/Users/gonciarz/Documents/MOSAIC/work/repo/RecursiveGPU/APRBench/Matlab');

% name='asdf.h5';
analysis_root='/Users/gonciarz/Documents/MOSAIC/work/repo/RecursiveGPU/build/';

xx=figure(1);
clf;
hold on;
format_figure(xx);
% plotData([analysis_root, 'BenchmarkLocalIntensityScaleTestOffset2.h5'], 1, 0);
% plotData([analysis_root, 'BenchmarkLocalIntensityScaleTestOffset6.h5'], 1, 1);
plotData(['../BenchmarkResults/v100/', 'BenchmarkLocalIntensityScaleTestOffset2.h5'], 1, 0);
plotData(['../BenchmarkResults/v100/', 'BenchmarkLocalIntensityScaleTestOffset6.h5'], 1, 1);
plotData(['../BenchmarkResults/Particulator/', 'BenchmarkLocalIntensityScaleTestOffset2.h5'], 1, 0);
plotData(['../BenchmarkResults/Particulator/', 'BenchmarkLocalIntensityScaleTestOffset6.h5'], 1, 1);

title('Local Intensity Scale Titan X vs 10 x Xeon(R)@2.6GHz')
l = legend({'CPU offset=2', 'GPU offset=2', 'CPU offset = 6', 'GPU offset = 6'});
l.Location = 'northwest';
l.Box = 'off';
l.FontSize = 20;
print('localIntensityScaleCpuVsGpu.eps' ,'-depsc','-painters','-loose','-cmyk');

xx=figure(2);
clf;
hold on;
format_figure(xx);
plotData(['../BenchmarkResults/v100/', 'BenchmarkBsplineTest.h5'], 0, 0);

plotData(['../BenchmarkResults/Particulator/', 'BenchmarkBsplineTest.h5'], 1, 1);

title('Recursive filter Titan X vs 10 x Xeon(R)@2.6GHz')
print('recursiveCpuVsGpu.eps' ,'-depsc','-painters','-loose','-cmyk');

function plotData(fileName, plotNum, colorShift)
    ad = load_analysis_data(fileName);
    
    [cpuData, cpuErr]=getMeanMeasurements(ad.CpuTime, ad.numOfRepetitions, ad.numOfRepetitionsToSkip);
    [gpuData, gpuErr]=getMeanMeasurements(ad.GpuDeviceTimeFull, ad.numOfRepetitions, ad.numOfRepetitionsToSkip);

    x=ad.ticksValue;
    
    cm_type = 'parula(5)';
    cm = colormap(cm_type)
    errorbar(x, cpuData,cpuErr, 'color', cm(3 + colorShift,:));
    errorbar(x, gpuData,gpuErr, 'color', cm(1 + colorShift,:));
    
%     axis([0 Inf 0 Inf]);    
    set(gca,'XTick', ad.ticksValue)
    set(gca,'XTickLabel', num2str(ad.ticksValue/ad.xNormalizer,strcat('%.',num2str(ad.numberOfDecimalPointsX),'f')));
    l = legend({'CPU', 'GPU'});
    l.Location = 'best';
    l.Box = 'off';
    l.FontSize = 20;
   
    xlabel(ad.xTitle');
    ylabel(ad.yTitle');            
    
    title(ad.plotTitle');    
end

function [out, maxErr]=getMeanMeasurements(data, noOfRep, skipSteps)
    numOfOutElements = length(data) / noOfRep;
    out = zeros(numOfOutElements,1);
    maxErr = zeros(numOfOutElements,1);
    i = 1;
    for idx=1:noOfRep:length(data)-(noOfRep-1)
        out(i) = mean(data(idx + skipSteps : idx + noOfRep - 1));
        maxErr(i) = std(data(idx + skipSteps:idx+noOfRep - 1));
        i = i + 1;
    end
end

function [out, maxErr]=getMinMeasurements(data, noOfRep, skipSteps)
    numOfOutElements = length(data) / noOfRep;
    out = zeros(numOfOutElements,1);
    maxErr = zeros(numOfOutElements,1);
    i = 1;
    for idx=1:noOfRep:length(data)-(noOfRep-1)
        out(i) = min(data(idx + skipSteps : idx + noOfRep - 1));
        maxErr(i) = std(data(idx + skipSteps:idx+noOfRep - 1));
        i = i + 1;
    end
end

end
