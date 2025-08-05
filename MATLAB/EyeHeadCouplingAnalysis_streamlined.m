function EyeHeadCouplingAnalysis_streamlined()
% created by EK 7/1/25
main_dir =  "C:\Users\emk263\Erin\Data\Eye_head_coupling";
data_directory = fullfile(main_dir,'rat_data\');
save_directory = fullfile(main_dir,'Analysis\analysis_testingnewmodel_rat');

single_filename = "";
% single_filename = "Tsh001_2025-06-11T12_50_45";
overwrite = 1; %0=create a new directory, 1=put files in existing directory, possibly overwriting old ones

plotting_params.make_saccade_processing_plots = 0; % plots showing the dlc/saccade data at different steps in the cleaning/preprocessing process
plotting_params.make_saccade_summary_plots = 1; % plots summarizing all detected saccades across the session
plotting_params.make_imu_summary_plots = 0; % plots summarizing IMU over time
plotting_params.make_full_session_zoomedin_plots = 0; % imu summary plot for each 30 second section in the recording
plotting_params.make_imu_videos = 0; % imu videos showing IMU and saccade
plotting_params.make_saccade_videos = 0; %imu/saccade videos but without imu and showing both eyes
plotting_params.plot_random_subset = 1; % only make plots for a subset of saccades
plotting_params.num_saccades_to_plot = 10;


%% general code structure
% set parameters
% generate save directory names and get foldernames for all the data folders
% for each file
    % load/format data
        % generate data filenames for this folder
        % make sure all files exist/have the right format
    % detect saccades and make saccade table
        
    % make saccade plots
    % categorize imu
    % plot imu data
    % make summary plots per day
    % generate saccade/imu table per day
% concatenate mass table of all saccade/imu information for all sessions in folder

    %% set parameters
    close all
    clc
    warning('off','MATLAB:illConditionedMatrix')   

    % parameters.framerate = 60;
    parameters.framerate = 61; %EK changed 7/1/25, this is roughly the average across all the sessions
    parameters.pupil_likelihood_thresh = 0.7;
    parameters.eye_likelihood_thresh = 0.7;
    parameters.outlier_stddev_thresh = 20;
    parameters.outlier_thresh = 40;
    parameters.smoothing_window = 9;
    %parameters.saccade_thresh = 15;
    % parameters.saccade_thresh = 15/parameters.pix2deg_calibration;
    parameters.saccade_thresh = 1; %EK changed 7/7/25 since it seemed to be missing a lot of saccades
    parameters.combine_saccade_thresh = parameters.saccade_thresh/2;
    parameters.rotate_eye_along_pca1 = 0;
    parameters.separate_licking = 0;
    % parameters.dlc_model_name = "DLC_resnet50_EyeSep30shuffle1_1000000"; %old eye model
    % parameters.dlc_model_name = "DLC_resnet50_EyeSep30shuffle1_1500000"; %new eye model as of 11/20/24
    % parameters.dlc_model_name = "DLC_resnet50_EyeSep30shuffle1_1800000"; %new eye model as of 5/3/25
    parameters.dlc_model_name = "DLC_resnet50_EyeSep30shuffle1_1920000"; %new eye model as of 7/25/25

    % tree shrew parameters
    tsh_parameters.pix2deg_calibration = 4;
    tsh_parameters.use_eye_centered = 0;
    tsh_parameters.filter_lower_x = 7; %not seeing any movement on this axis in TS data so we'll just use same as rats
    tsh_parameters.filter_higher_x = 12;
    tsh_parameters.filter_lower_y = 10;
    tsh_parameters.filter_higher_y = 13;
    tsh_parameters.filter_lower_z = 12;
    tsh_parameters.filter_higher_z = 20;

    % rat parameters
    rat_parameters.pix2deg_calibration = 5;
    rat_parameters.use_eye_centered = 1;
    rat_parameters.filter_lower_x = 7;
    rat_parameters.filter_higher_x = 12;
    rat_parameters.filter_lower_y = 10;
    rat_parameters.filter_higher_y = 18;
    rat_parameters.filter_lower_z = 10;
    rat_parameters.filter_higher_z = 18;



    %% generate save directory names and get foldernames for all the data sessions
    original_save_directory = save_directory;
    i=2;
    while isfolder(save_directory) & ~overwrite
        save_directory = strcat(original_save_directory,num2str((i)),'\');
        i=i+1;
    end
    figures_save_directory = fullfile(save_directory,'Figures\');
    saccade_processing_save_directory = fullfile(figures_save_directory,'saccade_processing_plots\'); 
    summary_plots_save_directory = fullfile(figures_save_directory,'saccade_summary_plots\');
    imu_summary_save_directory = fullfile(figures_save_directory,'imu_summary_plots\');
    full_session_zoomedin_plots_save_directory = fullfile(figures_save_directory,'full_session_zoomedin_plots\');
    imu_video_save_directory = fullfile(figures_save_directory,'imu_videos\');
    imu_video_save_directory_licks = fullfile(figures_save_directory,'imu_videos_licks\');
    saccade_video_save_directory = fullfile(figures_save_directory,'saccade_videos\');
    saccade_imu_table_fname = fullfile(save_directory,'saccade_imu_population_table');
    dlc_likelihood_table_fname = fullfile(save_directory,'dlc_likelihood_table');
    
    if ~isfolder(save_directory)
        mkdir(save_directory)
    end
    if ~isfolder(figures_save_directory)
        mkdir(figures_save_directory)
    end

    % get all foldernames for Rat data sessions
    data_folders = dir(data_directory);
    data_folders = struct2table(data_folders);
    data_foldernames = string(data_folders.name);
    data_folders = data_folders((contains(data_foldernames,'Rat')|contains(data_foldernames,'Tsh')),:);
    n_folders = height(data_folders);

    if single_filename ~= ""
        n_folders = 1;
    end

    dlc_likelihood_table = struct();
    dlc_likelihood_table.likelihoods = zeros(1,24);
    filecount = 0;

    %% for each session
    for foldernum = 1:n_folders
        close all
        % generate all data filenames for this folder
        % try
            if single_filename ~= ""
                filename = single_filename;
            else
                filename = string(data_folders.name(foldernum));
            end
            
            if contains(filename,"Tsh")
                parameters.pix2deg_calibration = tsh_parameters.pix2deg_calibration;
                parameters.use_eye_centered = tsh_parameters.use_eye_centered;
                parameters.filter_lower_x = tsh_parameters.filter_lower_x;
                parameters.filter_higher_x = tsh_parameters.filter_higher_x;
                parameters.filter_lower_y = tsh_parameters.filter_lower_y;
                parameters.filter_higher_y = tsh_parameters.filter_higher_y;
                parameters.filter_lower_z = tsh_parameters.filter_lower_z;
                parameters.filter_higher_z = tsh_parameters.filter_higher_z;
                parameters.dlc_model_name = "DLC_resnet50_EyeSep30shuffle1_1920000"; %new eye model as of 7/25/25
            elseif contains(filename,"Rat")
                parameters.pix2deg_calibration = rat_parameters.pix2deg_calibration;
                parameters.use_eye_centered = rat_parameters.use_eye_centered;
                parameters.filter_lower_x = rat_parameters.filter_lower_x;
                parameters.filter_higher_x = rat_parameters.filter_higher_x;
                parameters.filter_lower_y = rat_parameters.filter_lower_y;
                parameters.filter_higher_y = rat_parameters.filter_higher_y;
                parameters.filter_lower_z = rat_parameters.filter_lower_z;
                parameters.filter_higher_z = rat_parameters.filter_higher_z;
                parameters.dlc_model_name = "DLC_resnet50_EyeSep30shuffle1_1500000"; %new eye model as of 11/20/24
            end

            filename = filename{1}
            filecount = filecount+1;
            animalid = filename(1:6);
            timestamp = filename(8:26);
            date = datetime(str2num(timestamp(1:4)),str2num(timestamp(6:7)),str2num(timestamp(9:10)),str2num(timestamp(12:13)),str2num(timestamp(15:16)),str2num(timestamp(18:19)));
            fpath = strcat(data_directory,filename,'\');
            eye_left_dlc_fname = fullfile(fpath,strcat(animalid,"_E_L_",timestamp(1:end-5),"??_??",parameters.dlc_model_name,".csv"));
            eye_right_dlc_fname = fullfile(fpath,strcat(animalid,"_E_R_",timestamp(1:end-5),"??_??",parameters.dlc_model_name,".csv"));
            imu_fname = fullfile(fpath,strcat(animalid,"_IMU_",timestamp(1:end-5),"??_??",".csv"));
            camera_fname = fullfile(fpath,strcat(animalid,"_CameraLogger_",timestamp(1:end-5),"??_??",".csv"));
            licks_fname = fullfile(fpath,strcat(animalid,"_JuiceLogger_",timestamp(1:end-5),"??_??",".csv"));
            eye_left_vid_fname = fullfile(fpath,strcat(animalid,"_E_L_",timestamp(1:end-5),"??_??",parameters.dlc_model_name,"_labeled.mp4"));
            eye_right_vid_fname = fullfile(fpath,strcat(animalid,"_E_R_",timestamp(1:end-5),"??_??",parameters.dlc_model_name,"_labeled.mp4"));
            head_left_vid_fname = fullfile(fpath,strcat(animalid,"_H_L_",timestamp(1:end-5),"??_??",".mp4"));
            head_right_vid_fname = fullfile(fpath,strcat(animalid,"_H_R_",timestamp(1:end-5),"??_??",".mp4"));
    
            eye_left_dlc_fname = dir(eye_left_dlc_fname);
            eye_left_dlc_fname = strcat(fpath,eye_left_dlc_fname(1).name);
            eye_right_dlc_fname = dir(eye_right_dlc_fname);
            eye_right_dlc_fname = strcat(fpath,eye_right_dlc_fname(1).name);
            imu_fname = dir(imu_fname);
            imu_fname = strcat(fpath,imu_fname(1).name);
            camera_fname = dir(camera_fname);
            camera_fname = strcat(fpath,camera_fname(1).name);
            licks_fname = dir(licks_fname);
            licks_fname = strcat(fpath,licks_fname(1).name);
            eye_left_vid_fname = dir(eye_left_vid_fname);
            eye_left_vid_fname = strcat(fpath,eye_left_vid_fname(1).name);
            eye_right_vid_fname = dir(eye_right_vid_fname);
            eye_right_vid_fname = strcat(fpath,eye_right_vid_fname(1).name);
            if plotting_params.make_imu_videos
                head_left_vid_fname = dir(head_left_vid_fname);
                head_left_vid_fname = strcat(fpath,head_left_vid_fname(1).name);
                head_right_vid_fname = dir(head_right_vid_fname);
                head_right_vid_fname = strcat(fpath,head_right_vid_fname(1).name);
            end
    
    
            % make sure all files exist/have the right format
            if ~isfile(eye_left_dlc_fname)
                disp(strcat("No left eye DLC file found (with filename ",eye_left_dlc_fname,")"))
                continue;
            end
            if ~isfile(eye_right_dlc_fname)
                disp(strcat("No right eye DLC file found (with filename ",eye_right_dlc_fname,")"))
                continue;
            end
            if date < datetime(2024,9,30)
                disp("This recording is from before 9/30/2024 so the IMU data is not accurately synced with the videos")
            end
            if ~isfile(imu_fname)
                disp(strcat("No IMU file found (with filename ",imu_fname,")"))
            end
            if isempty(readtable(imu_fname,'NumHeaderLines',1))
                disp("IMU file is empty")
                continue;
            end
            if ~isfile(camera_fname)
                disp(strcat("No CameraLogger file found (with filename ",camera_fname,")"))
                continue;
            end
            if ~isfile(licks_fname)
                disp(strcat("No lick sensor file found (with filename ",licks_fname,")"))
                continue;
            end
            if isempty(readtable(licks_fname,'NumHeaderLines',1))
                disp("Lick sensor file is empty")
                continue;
            end
            if dateshift(date,'start','day') == dateshift(datetime(2024,11,13),'start', 'day')
                disp("Lick file is weird for this day")
                continue;
            end
            if ~isfile(eye_left_vid_fname) && (plotting_params.make_imu_videos)
                disp(strcat("No left eye DLC video file found (with filename ",eye_left_vid_fname,")"))
                eye_left_vid_fname = fullfile(fpath,strcat(animalid,"_E_L_",timestamp,".avi"));
                if ~isfile(eye_left_vid_fname)
                    disp(strcat("No left eye video file found (with filename ",eye_left_vid_fname,")"))
                    continue;
                end
            end
            if ~isfile(eye_right_vid_fname) && (plotting_params.make_imu_videos)
                disp(strcat("No right eye DLC video file found (with filename ",eye_right_vid_fname,")"))
                eye_right_vid_fname = fullfile(fpath,strcat(animalid,"_E_R_",timestamp,".avi"));
                if ~isfile(eye_right_vid_fname)
                    disp(strcat("No right eye video file found (with filename ",eye_right_vid_fname,")"))
                    continue;
                end
            end
            eye_left_vidObj = VideoReader(eye_left_vid_fname);
            eye_right_vidObj = VideoReader(eye_right_vid_fname);
            if ~isfile(head_left_vid_fname) && (plotting_params.make_imu_videos)
                disp(strcat("No left head video file found (with filename ",head_left_vid_fname,")"))
            end
            if ~isfile(head_right_vid_fname) && (plotting_params.make_imu_videos)
                disp(strcat("No right head video file found (with filename ",head_right_vid_fname,")"))
            end
            if plotting_params.make_imu_videos
                head_left_vidObj = VideoReader(head_left_vid_fname);
                head_right_vidObj = VideoReader(head_right_vid_fname);
            end
    
            temp_video = VideoReader(eye_left_vid_fname);
            eye_video_x_dim = temp_video.Width;
            eye_video_y_dim = temp_video.Height;
    
            %% load/format data (eye, IMU, licking)
            % load/format left/right eye DLC data (pose data from dlc = 8 points around pupil and 4 around eye)
            L_T = readtable(eye_left_dlc_fname,'NumHeaderLines',2);
            R_T = readtable(eye_right_dlc_fname,'NumHeaderLines',2);
    
            % load/format imu data (IMU = x/y/z accelerometer and x/y/z gyroscope)
            imu = readtable(imu_fname,'NumHeaderLines',1);
            % format imu data in the same way regardless of what order it was recorded in
            if size(imu,2)==10 %newer IMU files have time, accel x/y/z, gyro x/y/z and mag x/y/z, older files have accel x/y/z, gyro x/y/z, time
                imu_ts = table2array(imu(2:end,1));
                imu=table2array(imu(2:end,2:end));
            else
                imu_ts = table2array(imu(2:end,7));
                imu=table2array(imu(2:end,1:end-1));
            end
    
            % load cameraLogger file (for converting lick files to same timescale as DLC files
            cameraT = readtable(camera_fname,"NumHeaderLines",1);
            %calculate framerate from cameraLogger file
            % parameters.framerate = 1/mean(diff(table2array(cameraT(:,2))));
            % fprintf("Frame rate calculated from cameraLogger file = %2.2f\n",parameters.framerate)
            % if parameters.framerate>70
            %     fprintf("Frame rate is weird, skipping this file\n")
            %     continue;
            % end
    
            cameraT = table2array(cameraT(:,1));
    
    
            if date < datetime(2024,10,31) %camera frames used to be 1 off
                L_T = L_T(1:end-1,:);
                R_T = R_T(2:end,:);
                imu = imu(1:end-1,:);
                cameraT = cameraT(1:end-1,:);
            end
    
            % make sure that left and right eye DLC files are the same length
            if ~(height(L_T)==height(R_T))
                fprintf("Eye DLC files do not have the same number of data points (left = %d, right = %d)\n",height(L_T),height(R_T))
                if height(L_T)==height(R_T)+1
                    L_T = L_T(1:end-1,:);
                    disp("Removing last data point from left eye DLC")
                elseif height(R_T)==height(L_T)+1
                    R_T = R_T(1:end-1,:);
                    disp("Removing last data point from right eye DLC")
                else
                    disp("Skipping this file because eye DLC files are more than one off")
                    continue;
                end
            end
        
            L_T=table2array(L_T(:,2:end));
            R_T=table2array(R_T(:,2:end));
            T = zeros(length(L_T),36,2);
            T(:,:,1) = L_T;
            T(:,:,2) = R_T;
    
    
    
            % make sure that DLC, IMU and cameraLogger files are the same length
            if length(T)~=length(imu) | length(T)~=length(cameraT)
                fprintf("All files do not have the same number of data points (eye DLC = %d, IMU = %d, cameraLogger = %d)\n",length(T),length(imu),length(cameraT))
                minlength = min([length(T),length(imu),length(cameraT)]);
                fprintf("Removing %d extra data points from eye DLC files, continuing\n",length(T)-minlength)
                fprintf("Removing %d extra data points from IMU file, continuing\n",length(imu)-minlength)
                fprintf("Removing %d extra data points from cameraLogger file, continuing\n",length(cameraT)-minlength)
                T=T(1:minlength,:,:);
                imu=imu(1:minlength,:);
                cameraT=cameraT(1:minlength);
            end
            
            imu = imu-mean(imu,1);
    
            % load/format licking data (licking = licking onset/offset times as TRUE/FALSE)
            licktable = readtable(licks_fname,'NumHeaderLines',1);
            licks_idx = table2array(licktable(:,1));
            licks_TF=table2array(licktable(:,3));
            licks_frame = zeros(length(licks_idx),1);
            for i = 1:length(licks_idx)
                [minval,closestidx]=min(abs(cameraT-licks_idx(i)));
                licks_frame(i)=closestidx;
            end
            % convert true/false values to licking at each time point
            licking = zeros(length(T),1);
            for i = 1:length(licks_TF)-1
                if strcmp(licks_TF(i), 'True')
                    if licks_frame(i+1)+9<=length(licking) && licks_frame(i)-10>=1
                        licking(licks_frame(i)-10:licks_frame(i+1)+9) = 1;
                    elseif licks_frame(i)-10<1
                        licking(licks_frame(1):licks_frame(i+1)+9) = 1;
                    else
                        licking(licks_frame(i)-10:licks_frame(end)) = 1;
                    end
                end
            end
            licking = licking(1:length(T));
    
            
    
            ts=0:1/parameters.framerate:(length(T)-1)/parameters.framerate; %set up timestamps (in sec) to use for plotting
            ts_mins=ts/parameters.framerate; %set up timestamps (in min) to use for plotting
        % catch
        %     disp(strcat("Something went wrong with loading the files/data from ",filename,")"))
        %     continue;
        % end

        %% start saccade processing/detection

        if plotting_params.make_saccade_processing_plots % make plots showing the dlc/saccade data at different steps in the cleaning/preprocessing process
            if ~isfolder(saccade_processing_save_directory)
                mkdir(saccade_processing_save_directory)
            end

            %%%make saccade processing plot 1 - likelihoods for each keypoint
            fig=figure('position',[0         0        1800         1000]);
            subplot(2,1,1); hold on;
            sgtitle(strcat('Raw likelihoods (',filename,')'),"Interpreter","none");
            for i=3:3:36
                plot(ts_mins,T(:,i,1)+(i-3)/3)
            end
            title("Left eye")
            legend("anterior_eye","dorsal_eye","posterior_eye","ventral_eye","anterior_pupil","anterior_dorsal_pupil","dorsal_pupil","posterior_dorsal_pupil","posterior_pupil","posterior_ventral_pupil","ventral_pupil","anterior_ventral_pupil","Interpreter","none")
            ylabel("DLC keypoint likelihood (shifted for plotting)")
            xlim([0 max(ts_mins)])
        
            subplot(2,1,2); hold on;
            sgtitle(strcat('Raw likelihoods (',filename,')'),"Interpreter","none");
            for i=3:3:36
                plot(ts_mins,T(:,i,2)+(i-3)/3)
            end
            title("Right eye")
            xlabel("Time (min)")
            ylabel("DLC keypoint likelihood (shifted for plotting)")
            xlim([0 max(ts_mins)])
            savefile = strcat(saccade_processing_save_directory,filename,'_plot1_likelihood_plot.png');
            saveas(fig,savefile)       
        
            %grab points for eye and pupil
            x_eye = T(:,1:3:10,:);
            x_pupil = T(:,13:3:34,:);
            y_eye = T(:,2:3:11,:);
            y_pupil = T(:,14:3:35,:);
        
            % flip x coordinates for right eye video back to real coordinates
            x_pupil(:,:,2) = eye_video_x_dim/2 - x_pupil(:,:,2) + eye_video_x_dim/2;
            x_eye(:,:,2) = eye_video_x_dim/2 - x_eye(:,:,2) + eye_video_x_dim/2;
            
            %%%make saccade processing plots 3&6 - raw eye signal (each eye separately)
            fig=figure('position',[0         0        1800         1000]); 
            subplot(5,1,1);plot(ts_mins,x_eye(:,:,1));%legend("anterior_eye_x","dorsal_eye_x","posterior_eye_x","ventral_eye_x","Interpreter","none");title("Raw eye x")
            xlim([0 max(ts_mins)])
            title("Eye x positions")
            subplot(5,1,2);plot(ts_mins,y_eye(:,:,1));%legend("anterior_eye_y","dorsal_eye_y","posterior_eye_y","ventral_eye_y","Interpreter","none");title("Raw eye y")
            xlim([0 max(ts_mins)])
            title("Eye y positions")
            ylabel("Pixels")
            subplot(5,1,3);plot(ts_mins,x_pupil(:,:,1));%legend("anterior_pupil_x","anterior_dorsal_pupil_x","dorsal_pupil_x","posterior_dorsal_pupil_x","posterior_pupil_x","posterior_ventral_pupil_x","ventral_pupil_x","anterior_ventral_pupil_x","Interpreter","none");title("Raw pupil x")
            xlim([0 max(ts_mins)])
            title("Pupil x positions")
            subplot(5,1,4);plot(ts_mins,y_pupil(:,:,1));%legend("anterior_pupil_y","anterior_dorsal_pupil_y","dorsal_pupil_y","posterior_dorsal_pupil_y","posterior_pupil_y","posterior_ventral_pupil_y","ventral_pupil_y","anterior_ventral_pupil_y","Interpreter","none");title("Raw pupil y")
            xlim([0 max(ts_mins)])
            title("Pupil y positions")
            subplot(5,1,5); hold on;
            title("Lick sensor")
            plot(ts_mins,licking);
            xlabel('Time (min)');
            xlim([0 max(ts_mins)])
            ylim([0 1]);
            sgtitle(strcat('Raw eye signal - Left eye (',filename,')'),"Interpreter","none");
            savefile = strcat(saccade_processing_save_directory,filename,'_plot3_raw_eye_L.png');
            saveas(fig,savefile)  
        
            fig=figure('position',[0         0        1800         1000]); 
            subplot(5,1,1);plot(ts_mins,x_eye(:,:,2));%legend("anterior_eye_x","dorsal_eye_x","posterior_eye_x","ventral_eye_x","Interpreter","none");title("Raw eye x")
            xlim([0 max(ts_mins)])
            title("Eye x positions")
            subplot(5,1,2);plot(ts_mins,y_eye(:,:,2));%legend("anterior_eye_y","dorsal_eye_y","posterior_eye_y","ventral_eye_y","Interpreter","none");title("Raw eye y")
            xlim([0 max(ts_mins)])
            title("Eye y positions")
            ylabel("Pixels")
            subplot(5,1,3);plot(ts_mins,x_pupil(:,:,2));%legend("anterior_pupil_x","anterior_dorsal_pupil_x","dorsal_pupil_x","posterior_dorsal_pupil_x","posterior_pupil_x","posterior_ventral_pupil_x","ventral_pupil_x","anterior_ventral_pupil_x","Interpreter","none");title("Raw pupil x")
            xlim([0 max(ts_mins)])
            title("Pupil x positions")
            subplot(5,1,4);plot(ts_mins,y_pupil(:,:,2));%legend("anterior_pupil_y","anterior_dorsal_pupil_y","dorsal_pupil_y","posterior_dorsal_pupil_y","posterior_pupil_y","posterior_ventral_pupil_y","ventral_pupil_y","anterior_ventral_pupil_y","Interpreter","none");title("Raw pupil y")
            xlim([0 max(ts_mins)])
            title("Pupil y positions")
            subplot(5,1,5); hold on;
            title("Lick sensor")
            plot(ts_mins,licking);
            xlabel('Time (min)');
            xlim([0 max(ts_mins)])
            ylim([0 1]);
            sgtitle(strcat('Raw eye signal - Right eye (',filename,')'),"Interpreter","none");
            savefile = strcat(saccade_processing_save_directory,filename,'_plot6_raw_eye_R.png');
            saveas(fig,savefile) 
        end
        
        means_l = zeros(12,1);
        for i=3:3:36
            means_l((i-3)/3+1)=mean(T(:,i,1));
        end
        if any(means_l(1:4) < 0.9) || any(means_l([5,6,7,8,9,10,12]) < 0.95) || any(means_l(11) < 0.9)
            disp("The DLC tracking data for the left eye does not meet the likelihood criteria")
        end

        means_r = zeros(12,1);
        for i=3:3:36
            means_r((i-3)/3+1)=mean(T(:,i,2));
        end
        if any(means_r(1:4) < 0.9) || any(means_r([5,6,7,8,9,10,12]) < 0.95) || any(means_r(11) < 0.9)
            disp("The DLC tracking data for the right eye does not meet the likelihood criteria")
        end
        
        dlc_likelihood_table.filename(filecount) = {filename};
        dlc_likelihood_table.likelihoods(filecount,:) = [means_l',means_r'];

        %% 1) remove low likelihood keypoints (set keypoints with likelihood below thresh to NaN)
        for k=1:2
            for i=15:3:36
                for j=1:size(T,1)
                    if T(j,i,k)<parameters.pupil_likelihood_thresh
                        T(j,i-2:i,k)=NaN; %if likelihood of a keypoint around the pupil is below threshold set just that point to NaN
                    end
                end
            end
        end
        for k=1:2
            for i=3:3:12
                for j=1:size(T,1)
                    if T(j,i,k)<parameters.eye_likelihood_thresh
                        T(j,:,k)=NaN; %if likelihood of any keypoint around the eye is below threshold set the entire row to NaN
                    end
                end
            end
        end
    
        
        % if plotting_params.make_saccade_processing_plots % make plots showing the dlc/saccade data at different steps in the cleaning/preprocessing process
        %     %%%make saccade processing plot 2 - likelihoods for each keypoint now that low likelihoods have been removed
        %     fig=figure('position',[0         0        1800         1000]);
        %     subplot(2,1,1); hold on;
        %     for i=3:3:36
        %         plot(ts_mins,T(:,i,1)+(i-3)/3)
        %     end
        %     sgtitle(strcat('Raw likelihoods with low likelihood points removed (',filename,')'),"Interpreter","none");
        %     title("Left eye")
        %     legend("anterior_eye","dorsal_eye","posterior_eye","ventral_eye","anterior_pupil","anterior_dorsal_pupil","dorsal_pupil","posterior_dorsal_pupil","posterior_pupil","posterior_ventral_pupil","ventral_pupil","anterior_ventral_pupil","Interpreter","none")
        %     ylabel("DLC keypoint likelihood (shifted for plotting)")
        %     xlim([0 max(ts_mins)])
        % 
        %     subplot(2,1,2); hold on;
        %     for i=3:3:36
        %         plot(ts_mins,T(:,i,2)+(i-3)/3)
        %     end
        %     sgtitle(strcat('Raw likelihoods with low likelihood points removed (',filename,')'),"Interpreter","none");
        %     title("Right eye")
        %     xlabel("Time (min)")
        %     ylabel("DLC keypoint likelihood (shifted for plotting)")
        %     xlim([0 max(ts_mins)])
        % 
        %     savefile = strcat(saccade_processing_save_directory,filename,'_plot2_likelihood_plot_withlowlikelihoodremoved.png');
        %     saveas(fig,savefile)       
        % 
        %     %grab points for eye and pupil
        %     x_eye = T(:,1:3:10,:);
        %     x_pupil = T(:,13:3:34,:);
        %     y_eye = T(:,2:3:11,:);
        %     y_pupil = T(:,14:3:35,:);
        % 
        %     % flip x coordinates for right eye video back to real coordinates
        %     x_pupil(:,:,2) = eye_video_x_dim/2 - x_pupil(:,:,2) + eye_video_x_dim/2;
        %     x_eye(:,:,2) = eye_video_x_dim/2 - x_eye(:,:,2) + eye_video_x_dim/2;
        % 
        %     %%%make saccade processing plots 4&7 - eye signal with low likelihood removed (each eye separately)
        %     fig=figure('position',[0         0        1800         1000]); 
        %     subplot(4,1,1);plot(ts_mins,x_eye(:,:,1));%legend("anterior_eye_x","dorsal_eye_x","posterior_eye_x","ventral_eye_x","Interpreter","none");title("Raw eye x")
        %     xlim([0 max(ts_mins)])
        %     title("Eye x positions")
        %     subplot(4,1,2);plot(ts_mins,y_eye(:,:,1));%legend("anterior_eye_y","dorsal_eye_y","posterior_eye_y","ventral_eye_y","Interpreter","none");title("Raw eye y")
        %     xlim([0 max(ts_mins)])
        %     title("Eye y positions")
        %     ylabel("Pixels")
        %     subplot(4,1,3);plot(ts_mins,x_pupil(:,:,1));%legend("anterior_pupil_x","anterior_dorsal_pupil_x","dorsal_pupil_x","posterior_dorsal_pupil_x","posterior_pupil_x","posterior_ventral_pupil_x","ventral_pupil_x","anterior_ventral_pupil_x","Interpreter","none");title("Raw pupil x")
        %     xlim([0 max(ts_mins)])
        %     title("Pupil x positions")
        %     subplot(4,1,4);plot(ts_mins,y_pupil(:,:,1));%legend("anterior_pupil_y","anterior_dorsal_pupil_y","dorsal_pupil_y","posterior_dorsal_pupil_y","posterior_pupil_y","posterior_ventral_pupil_y","ventral_pupil_y","anterior_ventral_pupil_y","Interpreter","none");title("Raw pupil y")
        %     xlim([0 max(ts_mins)])
        %     title("Pupil y positions")
        %     xlabel("Time (min)")
        %     sgtitle(strcat('Raw eye signal with low likelihood points removed - Left eye (',filename,')'),"Interpreter","none");
        %     savefile = strcat(saccade_processing_save_directory,filename,'_plot4_raw_eye_withlowlikelihoodremoved_L.png');
        %     saveas(fig,savefile)  
        % 
        %     fig=figure('position',[0         0        1800         1000]); 
        %     subplot(4,1,1);plot(ts_mins,x_eye(:,:,2));%legend("anterior_eye_x","dorsal_eye_x","posterior_eye_x","ventral_eye_x","Interpreter","none");title("Raw eye x")
        %     xlim([0 max(ts_mins)])
        %     title("Eye x positions")
        %     subplot(4,1,2);plot(ts_mins,y_eye(:,:,2));%legend("anterior_eye_y","dorsal_eye_y","posterior_eye_y","ventral_eye_y","Interpreter","none");title("Raw eye y")
        %     xlim([0 max(ts_mins)])
        %     title("Eye y positions")
        %     ylabel("Pixels")
        %     subplot(4,1,3);plot(ts_mins,x_pupil(:,:,2));%legend("anterior_pupil_x","anterior_dorsal_pupil_x","dorsal_pupil_x","posterior_dorsal_pupil_x","posterior_pupil_x","posterior_ventral_pupil_x","ventral_pupil_x","anterior_ventral_pupil_x","Interpreter","none");title("Raw pupil x")
        %     xlim([0 max(ts_mins)])
        %     title("Pupil x positions")
        %     subplot(4,1,4);plot(ts_mins,y_pupil(:,:,2));%legend("anterior_pupil_y","anterior_dorsal_pupil_y","dorsal_pupil_y","posterior_dorsal_pupil_y","posterior_pupil_y","posterior_ventral_pupil_y","ventral_pupil_y","anterior_ventral_pupil_y","Interpreter","none");title("Raw pupil y")
        %     xlim([0 max(ts_mins)])
        %     title("Pupil y positions")
        %     xlabel("Time (min)")
        %     sgtitle(strcat('Raw eye signal with low likelihood points removed - Right eye (',filename,')'),"Interpreter","none");
        %     savefile = strcat(saccade_processing_save_directory,filename,'_plot7_raw_eye_withlowlikelihoodremoved_R.png');
        %     saveas(fig,savefile)  
        % end
        
        %% 2) remove any remaining crazy outliers
        
        T = filloutliers(T,NaN,"movmedian",100,1,"ThresholdFactor",parameters.outlier_stddev_thresh);
        
        %grab points for eye and pupil
        x_eye = T(:,1:3:10,:);
        x_pupil = T(:,13:3:34,:);
        y_eye = T(:,2:3:11,:);
        y_pupil = T(:,14:3:35,:);
        
        % flip x coordinates for right eye video back to real coordinates
        x_pupil(:,:,2) = eye_video_x_dim/2 - x_pupil(:,:,2) + eye_video_x_dim/2;
        x_eye(:,:,2) = eye_video_x_dim/2 - x_eye(:,:,2) + eye_video_x_dim/2;
        
        if plotting_params.make_saccade_processing_plots % make plots showing the dlc/saccade data at different steps in the cleaning/preprocessing process
            %%%make saccade processing plots 5&8 - eye signal with outliers removed (each eye separately)
            fig=figure('position',[0         0        1800         1000]); 
            subplot(4,1,1);plot(ts_mins,x_eye(:,:,1));%legend("anterior_eye_x","dorsal_eye_x","posterior_eye_x","ventral_eye_x","Interpreter","none");title("Raw eye x")
            xlim([0 max(ts_mins)])
            title("Eye x positions")
            subplot(4,1,2);plot(ts_mins,y_eye(:,:,1));%legend("anterior_eye_y","dorsal_eye_y","posterior_eye_y","ventral_eye_y","Interpreter","none");title("Raw eye y")
            xlim([0 max(ts_mins)])
            title("Eye y positions")
            ylabel("Pixels")
            subplot(4,1,3);plot(ts_mins,x_pupil(:,:,1));%legend("anterior_pupil_x","anterior_dorsal_pupil_x","dorsal_pupil_x","posterior_dorsal_pupil_x","posterior_pupil_x","posterior_ventral_pupil_x","ventral_pupil_x","anterior_ventral_pupil_x","Interpreter","none");title("Raw pupil x")
            xlim([0 max(ts_mins)])
            title("Pupil x positions")
            subplot(4,1,4);plot(ts_mins,y_pupil(:,:,1));%legend("anterior_pupil_y","anterior_dorsal_pupil_y","dorsal_pupil_y","posterior_dorsal_pupil_y","posterior_pupil_y","posterior_ventral_pupil_y","ventral_pupil_y","anterior_ventral_pupil_y","Interpreter","none");title("Raw pupil y")
            xlim([0 max(ts_mins)])
            title("Pupil y positions")
            xlabel("Time (min)")
            sgtitle(strcat('Raw eye signal with outliers removed - Left eye (',filename,')'),"Interpreter","none");
            savefile = strcat(saccade_processing_save_directory,filename,'_plot5_raw_eye_outliersremoved_L.png');
            saveas(fig,savefile)  
        
            fig=figure('position',[0         0        1800         1000]); 
            subplot(4,1,1);plot(ts_mins,x_eye(:,:,2));%legend("anterior_eye_x","dorsal_eye_x","posterior_eye_x","ventral_eye_x","Interpreter","none");title("Raw eye x")
            xlim([0 max(ts_mins)])
            title("Eye x positions")
            subplot(4,1,2);plot(ts_mins,y_eye(:,:,2));%legend("anterior_eye_y","dorsal_eye_y","posterior_eye_y","ventral_eye_y","Interpreter","none");title("Raw eye y")
            xlim([0 max(ts_mins)])
            title("Eye y positions")
            ylabel("Pixels")
            subplot(4,1,3);plot(ts_mins,x_pupil(:,:,2));%legend("anterior_pupil_x","anterior_dorsal_pupil_x","dorsal_pupil_x","posterior_dorsal_pupil_x","posterior_pupil_x","posterior_ventral_pupil_x","ventral_pupil_x","anterior_ventral_pupil_x","Interpreter","none");title("Raw pupil x")
            xlim([0 max(ts_mins)])
            title("Pupil x positions")
            subplot(4,1,4);plot(ts_mins,y_pupil(:,:,2));%legend("anterior_pupil_y","anterior_dorsal_pupil_y","dorsal_pupil_y","posterior_dorsal_pupil_y","posterior_pupil_y","posterior_ventral_pupil_y","ventral_pupil_y","anterior_ventral_pupil_y","Interpreter","none");title("Raw pupil y")
            xlim([0 max(ts_mins)])
            title("Pupil y positions")
            xlabel("Time (min)")
            sgtitle(strcat('Raw eye signal with outliers removed - Right eye (',filename,')'),"Interpreter","none");
            savefile = strcat(saccade_processing_save_directory,filename,'_plot8_raw_eye_outliersremoved_R.png');
            saveas(fig,savefile)  
        end
        

        %% 3) get pupil center position and size (by fitting ellipse)
        n_samples = length(x_pupil);
        
        pupil_ellipse = cell(n_samples,2);
        pupil_size = zeros(n_samples,2);
        center_x = zeros(n_samples,2);
        center_y = zeros(n_samples,2);
        for k=1:2
            for i=1:n_samples
                x = x_pupil(i,:,k);
                y = y_pupil(i,:,k);
                idx = isnan(x) | isnan(y);
                x = x(~idx)';
                y = y(~idx)';
                %if anynan(x_pupil(i,:)) || anynan(y_pupil(i,:)) %don't fit ellipse if any of the points are NaN (meaning the likelihood was low)
                if length(x)<5 %don't fit ellipse if it's missing more than 3 points
                    pupil_ellipse{i,k}=NaN;
                    pupil_size(i,k)=NaN;
                    center_x(i,k)=NaN;
                    center_y(i,k)=NaN;
                else
                    ellipse = fit_ellipse(squeeze(x_pupil(i,:,k))',squeeze(y_pupil(i,:,k))');
                    pupil_ellipse{i,k} = fit_ellipse(squeeze(x_pupil(i,:,k))',squeeze(y_pupil(i,:,k))');
                    if ~isempty(ellipse)
                        if ~isempty(ellipse.a)
                            pupil_size(i,k) = pi*pupil_ellipse{i,k}.a*pupil_ellipse{i,k}.b;
                            center_x(i,k)=pupil_ellipse{i,k}.X0_in;
                            center_y(i,k)=pupil_ellipse{i,k}.Y0_in;
                        else
                            pupil_size(i,k) = NaN;
                            center_x(i,k)=NaN;
                            center_y(i,k)=NaN;
                        end
                    else
                        pupil_size(i,k) = NaN;
                        center_x(i,k)=NaN;
                        center_y(i,k)=NaN;
                    end
                end
            end
        end        
        
        %% 4) put pupil center into eye centered coordinates, remove remaining outliers and smooth data
        temp_center_x = center_x;
        temp_center_y = center_y;

        x_eye=x_eye(:,[1,3],:); %EK added 5/28/31 remove for rats?
        y_eye=y_eye(:,[1,3],:); %EK added 5/28/31 remove for rats?

        % calculate center of the eye by averaging the 4 points around the eye
        center_eye_x = squeeze(mean(x_eye,2,"omitnan"));
        center_eye_y = squeeze(mean(y_eye,2,"omitnan"));        
        
        % remove outliers outside a specified threshold beyond the mean
        center_x(center_x>mean(center_x,"omitnan")+parameters.outlier_thresh | center_x<mean(center_x,"omitnan")-parameters.outlier_thresh)= NaN;
        center_eye_x(center_eye_x>mean(center_eye_x,"omitnan")+parameters.outlier_thresh | center_eye_x<mean(center_eye_x,"omitnan")-parameters.outlier_thresh)= NaN;
        center_y(center_y>mean(center_y,"omitnan")+parameters.outlier_thresh | center_y<mean(center_y,"omitnan")-parameters.outlier_thresh)= NaN;
        center_eye_y(center_eye_y>mean(center_eye_y,"omitnan")+parameters.outlier_thresh | center_eye_y<mean(center_eye_y,"omitnan")-parameters.outlier_thresh)= NaN;

        center_x_rawcoords = center_x;
        center_y_rawcoords = center_y;

        smoothed_x = smoothdata(center_x,1,"movmedian",parameters.smoothing_window,"includenan");
        smoothed_y = smoothdata(center_y,1,"movmedian",parameters.smoothing_window,"includenan"); 

        smoothed_x_eye = (smoothed_x-center_eye_x)/parameters.pix2deg_calibration; % convert eye center position from pixels to degrees
        smoothed_y_eye = (smoothed_y-center_eye_y)/parameters.pix2deg_calibration; % convert eye center position from pixels to degrees
        smoothed_x_eye = smoothed_x_eye-mean(smoothed_x_eye,1,"omitnan");
        smoothed_y_eye = smoothed_y_eye-mean(smoothed_y_eye,1,"omitnan");

        smoothed_x_camera = (smoothed_x-mean(center_eye_x,1,"omitnan"))/parameters.pix2deg_calibration; % convert eye center position from pixels to degrees
        smoothed_y_camera = (smoothed_y-mean(center_eye_y,1,"omitnan"))/parameters.pix2deg_calibration; % convert eye center position from pixels to degrees
        smoothed_x_camera = smoothed_x_camera-mean(smoothed_x_camera,1,"omitnan");
        smoothed_y_camera = smoothed_y_camera-mean(smoothed_y_camera,1,"omitnan");

        smoothed_x_eye_oversampled = interp(smoothed_x_eye(:,1),2);
        smoothed_y_eye_oversampled = interp(smoothed_y_eye(:,1),2);
        smoothed_x_camera_oversampled = interp(smoothed_x_camera(:,1),2);
        smoothed_y_camera_oversampled = interp(smoothed_y_camera(:,1),2);

        %% 5) detect saccades (times where the distance the eye moved is above thresh)
        % calculate movements of the pupil relative to the camera
        dl_camera = sqrt((diff(smoothed_y_camera).^2)+(diff(smoothed_x_camera).^2));

        % calculate movements of the pupil relative to the eye
        dl_eye = sqrt((diff(smoothed_y_eye).^2)+(diff(smoothed_x_eye).^2));

        if parameters.use_eye_centered %this controls whether you want to use eye centered or camera centered coordinates for calculations/plotting
            dl = dl_eye; 
            smoothed_x = smoothed_x_eye;
            smoothed_y = smoothed_y_eye;
        else
            dl = dl_camera;
            smoothed_x = smoothed_x_camera;
            smoothed_y = smoothed_y_camera;
        end

        if parameters.rotate_eye_along_pca1
            try
                left_points = [smoothed_x(:,1) smoothed_y(:,1)];
                left_points = left_points-mean(left_points,1,"omitnan");
                [coeff_left,~,~]=pca(left_points);
                left_points= [smoothed_x(:,1) smoothed_y(:,1)]*coeff_left;
                atan2d(coeff_left(2,1),coeff_left(1,1))
                smoothed_x(:,1) = left_points(:,1);
                smoothed_y(:,1) = left_points(:,2);
            catch
                print("error with PCA/n")
            end
            try
                right_points = [smoothed_x(:,2) smoothed_y(:,2)];
                right_points = right_points-mean(right_points,1,"omitnan");
                [coeff_right,~,~]=pca(right_points);
                right_points= right_points*coeff_right;
                atan2d(coeff_right(2,1),coeff_right(1,1))
                smoothed_x(:,2) = right_points(:,1);
                smoothed_y(:,2) = right_points(:,2);
            catch
                print("error with PCA/n")
            end
        end

        t=wrapTo180(atan2d(diff(smoothed_y),diff(smoothed_x))+90);

        if plotting_params.make_saccade_processing_plots % make plots showing the dlc/saccade data at different steps in the cleaning/preprocessing process
            %%%make saccade processing plot 9 - putting pupil into eye coordinates
            fig=figure('position',[0         0        1800         1000]); 
            hold on;
            subplot(6,1,1);plot(ts_mins,center_eye_x);legend("left eye","right eye")
            title("Eye center X")
            xlim([0 max(ts_mins)])
            subplot(6,1,2);hold on;plot(ts_mins,smoothed_x_camera);
            title("Pupil center X (camera coordinates)")
            xlim([0 max(ts_mins)])
            subplot(6,1,3);plot(ts_mins,smoothed_x_eye);
            title("Eye centered pupil center (Pupil center X - eye center X)")
            xlim([0 max(ts_mins)])
            subplot(6,1,4);plot(ts_mins,center_eye_y);
            title("Eye center Y")
            ylabel("Pixels")
            xlim([0 max(ts_mins)])
            subplot(6,1,5);hold on;plot(ts_mins,smoothed_y_camera);
            title("Pupil center Y (camera coordinates)")
            xlim([0 max(ts_mins)])
            subplot(6,1,6);plot(ts_mins,smoothed_y_eye);
            title("Eye centered pupil center (Pupil center Y - eye center Y)")
            xlim([0 max(ts_mins)])
            xlabel("Time (min)")
            sgtitle(strcat('Putting pupil into eye-centered coordinates (',filename,')'),"Interpreter","none");
            savefile = strcat(saccade_processing_save_directory,filename,'_plot9_pupilcenter_ineyecoords.png');
            saveas(fig,savefile)  

            %%%make saccade processing plot 10 - pupil center before and after processing
            fig=figure('position',[0         0        1800         1000]); 
            subplot(4,1,1);
            plot(ts_mins,temp_center_x);legend("left eye","right eye")
            xlim([0 max(ts_mins)])
            title("Pupil center x (before)")
            ylabel("Pixels")
            subplot(4,1,2);
            plot(ts_mins,smoothed_x);
            xlim([0 max(ts_mins)])
            title("Pupil center x (after)")
            ylabel("Degrees")
            subplot(4,1,3);
            plot(ts_mins,temp_center_y);
            xlim([0 max(ts_mins)])
            title("Pupil center y (before)")
            ylabel("Pixels")
            subplot(4,1,4);
            plot(ts_mins,smoothed_y);
            xlim([0 max(ts_mins)])
            title("Pupil center y (after)")
            xlabel("Time (min)")
            ylabel("Degrees")
            sgtitle(strcat('Pupil center before and after processing (',filename,')'),"Interpreter","none");
            savefile = strcat(saccade_processing_save_directory,filename,'_plot10_pupilcenter_beforeafterprocessing.png');
            saveas(fig,savefile)  
        end

        ix_camera_cell = {};
        ix_eye_cell={};
        ix_cell = {};
        ix_start_cell = {};
        ix_end_cell = {};

        % find saccades - times where the distance moved (both relative to camera and relative to eye) is above thresh
        for k=1:2
            [vals,ix_raw]=findpeaks(dl_camera(:,k));
            ix_raw = ix_raw(vals>parameters.saccade_thresh);
            ix_camera_cell{k}=ix_raw;
        
            [vals,ix_eye]=findpeaks(dl_eye(:,k));
            ix_eye = ix_eye(vals>parameters.saccade_thresh);
            ix_eye_cell{k}=ix_eye;
        
            [ix,temp] = intersect(ix_raw, ix_eye); % only count saccades that occur when the eye moves both relative to the camera and relative to the eye
            ix_cell{k}=ix;

            ix_start = zeros(length(ix),1);
            ix_end = zeros(length(ix),1);
            count = 1;
            for i = ix'
                start_frame = i;
                while dl_eye(start_frame-1,k)>parameters.combine_saccade_thresh && dl_camera(start_frame-1,k)>parameters.combine_saccade_thresh
                    start_frame=start_frame-1;
                    if start_frame<1
                        break;
                    end
                end
                ix_start(count)=start_frame;
                end_frame = i;
                while dl_eye(end_frame,k)>parameters.combine_saccade_thresh && dl_camera(end_frame,k)>parameters.combine_saccade_thresh
                    end_frame=end_frame+1;
                    if end_frame>=length(dl_eye)-1
                        break;
                    end
                end
                ix_end(count)=end_frame;
                count=count+1;
            end
            ix_start_cell{k}=ix_start;
            ix_end_cell{k}=ix_end;
        end
        
        ix = unique([ix_cell{1}' ix_cell{2}']);

        ix = ix(ix<length(T)-round(parameters.framerate) & ix>round(parameters.framerate));
        
        % combine1 = find((dl(ix+1,1)>parameters.saccade_thresh/2 | dl(ix-1,1)>parameters.saccade_thresh/2)&(dl_raw(ix+1,1)>parameters.saccade_thresh/2 | dl_raw(ix-1,1)>parameters.saccade_thresh/2));
        % combine2 = find((dl(ix+1,2)>parameters.saccade_thresh/2 | dl(ix-1,2)>parameters.saccade_thresh/2)&(dl_raw(ix+1,2)>parameters.saccade_thresh/2 | dl_raw(ix-1,2)>parameters.saccade_thresh/2));
        % potential_saccades_to_combine = unique([combine1' combine2']);

        % for i = ix(potential_saccades_to_combine)
        %     figure('position',[400         400        1200         400]);hold on;
        %     subplot(2,3,1); hold on;
        %     plot(dl(i-10:i+10,1));
        %     plot(dl(i-10:i+10,2));
        %     yline(parameters.saccade_thresh)
        %     subplot(2,3,2); hold on;
        %     plot(smoothed_x(i-10:i+10,1));
        %     plot(smoothed_x(i-10:i+10,2));
        %     subplot(2,3,3); hold on;
        %     plot(smoothed_y(i-10:i+10,1));
        %     plot(smoothed_y(i-10:i+10,2));
        % 
        %     subplot(2,3,4); hold on;
        %     plot(dl_raw(i-10:i+10,1));
        %     plot(dl_raw(i-10:i+10,2));
        %     yline(parameters.saccade_thresh)
        %     subplot(2,3,5); hold on;
        %     plot(smoothed_x_rawcoords(i-10:i+10,1));
        %     plot(smoothed_x_rawcoords(i-10:i+10,2));
        %     subplot(2,3,6); hold on;
        %     plot(smoothed_y_rawcoords(i-10:i+10,1));
        %     plot(smoothed_y_rawcoords(i-10:i+10,2));
        %     yline(parameters.saccade_thresh)
        % 
        % end

        %% store information about saccades in saccade table
        if parameters.use_eye_centered %this controls whether you want to use eye centered or camera centered coordinates for calculations/plotting
            x_temp = smoothed_x_eye;
            y_temp = smoothed_y_eye;
        else
            x_temp = smoothed_x_camera;
            y_temp = smoothed_y_camera;
        end


        if isempty(ix)
            continue;
        end

        i=1;
        count=1;
        saccade_imu_table = table('Size',[length(ix) 0]);
        while count <= length(ix)
            idx = ix(count);

            if idx-round(parameters.framerate) < 0 || idx+round(parameters.framerate)>length(x_temp) %skip saccades that were too close to the start/end of the recording
                count=count+1;
                continue;
            end

            saccade_imu_table.filename(i) = {filename};
            saccade_imu_table.animal_id(i) = {animalid};
            saccade_imu_table.date(i) = date;
            saccade_imu_table.saccade_num(i)= i;
            saccade_imu_table.saccade_id(i) = {strcat(filename,'_saccade',num2str(i))};
            saccade_imu_table.saccade_frame(i) = idx;

            eye = 1;
            temp_idx = NaN;
            if ismember(idx,ix_cell{eye})
                temp_idx = idx;
            elseif ismember(idx+1,ix_cell{eye})
                temp_idx = idx+1;
            end

            if ~isnan(temp_idx)
                saccade_imu_table.left_idx(i) = temp_idx;
                saccade_imu_table.left_mag(i) = dl(temp_idx,eye);
                saccade_imu_table.left_dir(i) = t(temp_idx,eye);
                saccade_imu_table.left_ts(i) = ts_mins(temp_idx);
                eye_cell_idx = find(temp_idx == ix_cell{eye});
                start_idx = ix_start_cell{eye}(eye_cell_idx);
                end_idx = ix_end_cell{eye}(eye_cell_idx);
                saccade_imu_table.left_start_idx(i) = start_idx;
                saccade_imu_table.left_end_idx(i) = end_idx;
                saccade_imu_table.left_mag_comb(i) = sqrt(((y_temp(end_idx)-y_temp(start_idx)).^2)+((x_temp(end_idx)-x_temp(start_idx)).^2));
                saccade_imu_table.left_vel_comb(i) = sqrt(((y_temp(end_idx)-y_temp(start_idx)).^2)+((x_temp(end_idx)-x_temp(start_idx)).^2))/(ts(end_idx)-ts(start_idx));
                saccade_imu_table.left_dir_comb(i) = atan2d((y_temp(end_idx)-y_temp(start_idx)),(x_temp(end_idx)-x_temp(start_idx)));
                saccade_imu_table.left_dur_comb(i) = (ts(end_idx)-ts(start_idx))*1000;
            else
                saccade_imu_table.left_idx(i) = NaN;
                saccade_imu_table.left_mag(i) = dl(idx,eye);
                saccade_imu_table.left_dir(i) = t(idx,eye);
                saccade_imu_table.left_ts(i) = NaN;
                saccade_imu_table.left_start_idx(i) = NaN;
                saccade_imu_table.left_end_idx(i) = NaN;
                saccade_imu_table.left_mag_comb(i) = NaN;
                saccade_imu_table.left_vel_comb(i) = NaN;
                saccade_imu_table.left_dir_comb(i) = NaN;
                saccade_imu_table.left_dur_comb(i) = NaN;
            end

            eye = 2;
            temp_idx = NaN;
            if ismember(idx,ix_cell{eye})
                temp_idx = idx;
            elseif ismember(idx+1,ix_cell{eye})
                temp_idx = idx+1;
            end

            if ~isnan(temp_idx)
                saccade_imu_table.right_idx(i) = temp_idx;
                saccade_imu_table.right_mag(i) = dl(temp_idx,eye);
                saccade_imu_table.right_dir(i) = t(temp_idx,eye);
                saccade_imu_table.right_ts(i) = ts_mins(temp_idx);
                eye_cell_idx = find(temp_idx == ix_cell{eye});
                start_idx = ix_start_cell{eye}(eye_cell_idx);
                end_idx = ix_end_cell{eye}(eye_cell_idx);
                saccade_imu_table.right_start_idx(i) = start_idx;
                saccade_imu_table.right_end_idx(i) = end_idx;
                saccade_imu_table.right_mag_comb(i) = sqrt(((y_temp(end_idx)-y_temp(start_idx)).^2)+((x_temp(end_idx)-x_temp(start_idx)).^2));
                saccade_imu_table.right_vel_comb(i) = sqrt(((y_temp(end_idx)-y_temp(start_idx)).^2)+((x_temp(end_idx)-x_temp(start_idx)).^2))/(ts(end_idx)-ts(start_idx));
                saccade_imu_table.right_dir_comb(i) = atan2d((y_temp(end_idx)-y_temp(start_idx)),(x_temp(end_idx)-x_temp(start_idx)));
                saccade_imu_table.right_dur_comb(i) = (ts(end_idx)-ts(start_idx))*1000;
            else
                saccade_imu_table.right_idx(i) = NaN;
                saccade_imu_table.right_mag(i) = dl(idx,eye);
                saccade_imu_table.right_dir(i) = t(idx,eye);
                saccade_imu_table.right_ts(i) = NaN;
                saccade_imu_table.right_start_idx(i) = NaN;
                saccade_imu_table.right_end_idx(i) = NaN;
                saccade_imu_table.right_mag_comb(i) = NaN;
                saccade_imu_table.right_vel_comb(i) = NaN;
                saccade_imu_table.right_dir_comb(i) = NaN;
                saccade_imu_table.right_dur_comb(i) = NaN;
            end

            saccade_imu_table.interoccular_angle(i) = wrapTo180(saccade_imu_table.right_dir(i)-saccade_imu_table.left_dir(i));
            saccade_imu_table.timestamp(i) = ts_mins(idx);
            saccade_imu_table.framerate(i) = parameters.framerate;
            saccade_imu_table.session_duration(i) = max(ts_mins);
            saccade_imu_table.left_x(i) = {smoothed_x(idx-round(parameters.framerate):idx+round(parameters.framerate),1)};
            saccade_imu_table.left_y(i) = {smoothed_y(idx-round(parameters.framerate):idx+round(parameters.framerate),1)};
            saccade_imu_table.right_x(i) = {smoothed_x(idx-round(parameters.framerate):idx+round(parameters.framerate),2)};
            saccade_imu_table.right_y(i) = {smoothed_y(idx-round(parameters.framerate):idx+round(parameters.framerate),2)};

            i=i+1;
            count=count+1;
        end

        % saccade_imu_table = saccade_imu_table(abs(saccade_imu_table.interoccular_angle)>45,:);

        saccade_imu_table = movevars(saccade_imu_table,"right_idx",'After',"left_idx");
        saccade_imu_table = movevars(saccade_imu_table,"right_mag",'After',"left_mag");
        saccade_imu_table = movevars(saccade_imu_table,"right_dir",'After',"left_dir");
        saccade_imu_table = movevars(saccade_imu_table,"right_ts",'After',"left_ts");
        saccade_imu_table = rmmissing(saccade_imu_table,'DataVariables',"date");
        saccade_imu_table.sac_mag = max(saccade_imu_table{:,["left_mag","right_mag"]},[],2);
        ix = saccade_imu_table.saccade_frame;
    
        
        %% save imu info in saccade imu table

        stds = zeros(0,3);
        binsz=600;
        idxs = zeros(0);
        for i = 1:500
            idx = randi(length(imu)-binsz);
            idxs(i) = idx;
            bin = imu(idx:idx+binsz,1:3);
            stds(i,:)=std(bin);
        end

        median_std = median(stds);

        %%% Ratnadeep's algorithm 
        % median_std = [0.000670566383685815, 0.000639472666829974, 0.000844034025101936]; %median stdevs across all the recordings
        % median_std = [0.000430634819488357, 0.000385643857544995, 0.000524476925645335]; %median stdevs across all the recordings (5-20Hz filtered imu)

        if max(imu_ts)>1000000000
            disp("IMU timestamps are weird, skip this file")
            continue;
        end

        T = imu_ts(end) - imu_ts(1); % Total time
        N = length(imu_ts); % Number of samples
        fs = N * 1000 / T; % Sampling frequency in Hz

        % Bandpass filter the data
        % Design 10th-order bandpass filter between 10 and 15 Hz
        filtered_imu = zeros(length(imu),3);

        [sos, g] = butter(10, [parameters.filter_lower_x, parameters.filter_higher_x] / (fs / 2));
        filtered_imu(:,1) = filtfilt(sos, g, imu(:,1)); % Zero-phase filtering

        [sos, g] = butter(10, [parameters.filter_lower_y, parameters.filter_higher_y] / (fs / 2));
        filtered_imu(:,2) = filtfilt(sos, g, imu(:,2)); % Zero-phase filtering

        [sos, g] = butter(10, [parameters.filter_lower_z, parameters.filter_higher_z] / (fs / 2));
        filtered_imu(:,3) = filtfilt(sos, g, imu(:,3)); % Zero-phase filtering

        processed_imu = hilbert(filtered_imu); % Analytic signal
        processed_imu = abs(processed_imu); % Magnitude of analytic signal

        for i=1:length(ix)
            sac = ix(i);
            window = round(parameters.framerate);
            frames = sac-window:sac+window;

            saccade_imu_table.licking(i) = {licking(frames)'}; %licking 
            saccade_imu_table.lick_sum(i) = sum(saccade_imu_table.licking{i});
            saccade_imu_table.imu_envelope(i) = {(processed_imu(frames,:)./median_std)'};
        end

        if ~exist('saccade_imu_population_table', 'var')
            saccade_imu_population_table = saccade_imu_table;
        else
            saccade_imu_population_table = [saccade_imu_population_table; saccade_imu_table];
        end

        
        % figure;
        % plot(processed_imu);
        % lick_diff = diff(licking)';
        % licks_on = find(lick_diff==1);
        % licks_off = find(lick_diff==-1);
        % if ~isempty(licks_on) && ~isempty(licks_off)
        %     if isempty(licks_on) && isscalar(licks_off)
        %         licks_on = 1;
        %     end
        %     if isempty(licks_off) && isscalar(licks_on)
        %         licks_off = length(licking);
        %     end
        %     if licks_on(1)>licks_off(1)
        %         licks_on = [1 licks_on];
        %     end
        %     if licks_on(end)>licks_off(end)
        %         licks_off = [licks_off length(licking)];
        %     end
        % end
        % lick_plot_coords = [licks_on;licks_off];
        % yl = ylim;
        % y = yl(1);
        % h = yl(2) - yl(1);
        % 
        % for i=1:size(lick_plot_coords,2)
        %     x = lick_plot_coords(1,i);
        %     w = lick_plot_coords(2,i) - lick_plot_coords(1,i);
        %     if w<0
        %         continue;
        %     end
        %     rectangle('position', [x y w h],"FaceColor",'b',"FaceAlpha",0.2,'edgecolor', 'none');
        % end
        % 
        % licks_removed = processed_imu(licking==0,:);
        % licks_only = processed_imu(licking==1,:);
        % 
        % % figure;plot(licks_removed)
        % % figure;plot(licks_only)
        % mean(processed_imu)
        % mean(licks_removed)
        % mean(licks_only)
        % median(processed_imu)
        % median(licks_removed)
        % median(licks_only)
        % corr(processed_imu,licking)
        % test = licking(randperm(length(licking)));
        % corr(processed_imu,test)


        %% make saccade summary plots - plots summarizing all detected saccades across the session
        if plotting_params.make_saccade_summary_plots
            if ~isfolder(summary_plots_save_directory) 
                mkdir(summary_plots_save_directory)
            end

            %%%make saccade summary plot 1 - eye x and y position over time with detected saccades overlaid
            fig=figure('position',[0         0        1800         1000]); 
            sgtitle(strcat('Pupil X and Y position across time (',filename,')'),"Interpreter","none");
        
            subplot(5,1,1);hold on;
            title("Pupil x")
            plot(ts_mins,smoothed_x);
            plot(rmmissing(saccade_imu_table.left_ts),smoothed_x(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),smoothed_x(rmmissing(saccade_imu_table.right_idx),2),'k.')
            xlim([0 max(ts_mins)])
        
            subplot(5,1,2); hold on;
            title("Pupil y")
            plot(ts_mins,smoothed_y);
            plot(rmmissing(saccade_imu_table.left_ts),smoothed_y(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),smoothed_y(rmmissing(saccade_imu_table.right_idx),2),'k.')
            ylabel('Degrees');
            xlim([0 max(ts_mins)])
        
            subplot(5,1,3); hold on;
            title("Pupil x,y change (eye centered)")
            plot(ts_mins(2:end),dl_eye(:,1));
            plot(ts_mins(2:end),-dl_eye(:,2));
            plot(rmmissing(saccade_imu_table.left_ts),dl_eye(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),-dl_eye(rmmissing(saccade_imu_table.right_idx),2),'k.')
            xlim([0 max(ts_mins)])
        
            subplot(5,1,4); hold on;
            title("Pupil x,y change (raw pupil)")
            plot(ts_mins(2:end),dl_camera(:,1));
            plot(ts_mins(2:end),-dl_camera(:,2));
            plot(rmmissing(saccade_imu_table.left_ts),dl_camera(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),-dl_camera(rmmissing(saccade_imu_table.right_idx),2),'k.')
            xlabel('Time (min)');
            xlim([0 max(ts_mins)])
        
            subplot(5,1,5); hold on;
            title("Lick sensor")
            plot(ts_mins,licking);
            xlabel('Time (min)');
            xlim([0 max(ts_mins)])
            ylim([0 1]);

            savefile = strcat(summary_plots_save_directory,filename,'_sumplot1_x&y&saccades.png');
            saveas(fig,savefile)
        
            %%%make saccade summary plot 1a - eye x and y position over time with detected saccades overlaid (one minute zoomed in section)
            fig=figure('position',[0         0        1800         1000]); 
            sgtitle(strcat('Pupil X and Y position across time - one minute with most saccades zoomed in (',filename,')'),"Interpreter","none");
        
            most_saccades = 0;
            most_saccades_idx = 1;
            for i=0:floor(max(ts_mins)) %find the minute with the most saccades
                count_saccades = sum(ix>=i*parameters.framerate*60&ix<(i+1)*parameters.framerate*60);
                if count_saccades > most_saccades
                    most_saccades_idx = i;
                    most_saccades = count_saccades;
                end
            end

            subplot(5,1,1);hold on;
            title("Pupil x")
            plot(ts_mins,smoothed_x);
            plot(rmmissing(saccade_imu_table.left_ts),smoothed_x(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),smoothed_x(rmmissing(saccade_imu_table.right_idx),2),'k.')
            xlim([most_saccades_idx most_saccades_idx+1])
        
            subplot(5,1,2); hold on;
            title("Pupil y")
            plot(ts_mins,smoothed_y);
            plot(rmmissing(saccade_imu_table.left_ts),smoothed_y(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),smoothed_y(rmmissing(saccade_imu_table.right_idx),2),'k.')
            ylabel('Degrees');
            xlim([most_saccades_idx most_saccades_idx+1])
        
            subplot(5,1,3); hold on;
            title("Pupil x,y change (eye centered)")
            plot(ts_mins(2:end),dl_eye(:,1));
            plot(ts_mins(2:end),-dl_eye(:,2));
            plot(rmmissing(saccade_imu_table.left_ts),dl_eye(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),-dl_eye(rmmissing(saccade_imu_table.right_idx),2),'k.')
            xlim([most_saccades_idx most_saccades_idx+1])
        
            subplot(5,1,4); hold on;
            title("Pupil x,y change (raw pupil)")
            plot(ts_mins(2:end),dl_camera(:,1));
            plot(ts_mins(2:end),-dl_camera(:,2));
            plot(rmmissing(saccade_imu_table.left_ts),dl_camera(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),-dl_camera(rmmissing(saccade_imu_table.right_idx),2),'k.')
            xlabel('Time (min)');
            xlim([most_saccades_idx most_saccades_idx+1])
        
            subplot(5,1,5); hold on;
            title("Lick sensor")
            plot(ts_mins,licking);
            xlabel('Time (min)');
            xlim([most_saccades_idx most_saccades_idx+1])
            ylim([0 1]);

            savefile = strcat(summary_plots_save_directory,filename,'_sumplot1a_x&y&saccades_zoomedin.png');
            saveas(fig,savefile)

            if parameters.separate_licking
                %%%(if removing licks) make saccade summary plot 1b - eye x and y position over time with detected saccades overlaid (licks removed)
                fig=figure('position',[0         0        1800         1000]); 
                sgtitle(strcat('Pupil X and Y position across time - licks removed (',filename,')'),"Interpreter","none");
                [lick_idxs,val] = find(licking==1);
                smoothed_x_eye_nolicks = smoothed_x_eye;
                smoothed_x_eye_nolicks(lick_idxs,:) = NaN;
                smoothed_y_eye_nolicks = smoothed_y_eye;
                smoothed_y_eye_nolicks(lick_idxs,:) = NaN;
                dl_eye_nolicks = dl_eye;
                dl_eye_nolicks(lick_idxs,:) = NaN;
                dl_eye_nolicks = dl_eye_nolicks(1:length(dl_eye),:);
                dl_camera_nolicks = dl_camera;
                dl_camera_nolicks(lick_idxs,:) = NaN;
                dl_camera_nolicks = dl_camera_nolicks(1:length(dl_eye),:);
                subplot(5,1,1);hold on;
                title("Pupil x")
                plot(ts_mins,smoothed_x_eye_nolicks);
                plot(rmmissing(saccade_imu_table.left_ts),smoothed_x_eye_nolicks(rmmissing(saccade_imu_table.left_idx),1),'k.')
                plot(rmmissing(saccade_imu_table.right_ts),smoothed_x_eye_nolicks(rmmissing(saccade_imu_table.right_idx),2),'k.')
                xlim([0 max(ts_mins)])
        
                subplot(5,1,2); hold on;
                title("Pupil y")
                plot(ts_mins,smoothed_y_eye_nolicks);
                plot(rmmissing(saccade_imu_table.left_ts),smoothed_y_eye_nolicks(rmmissing(saccade_imu_table.left_idx),1),'k.')
                plot(rmmissing(saccade_imu_table.right_ts),smoothed_y_eye_nolicks(rmmissing(saccade_imu_table.right_idx),2),'k.')
                ylabel('Degrees');
                xlim([0 max(ts_mins)])
        
                subplot(5,1,3); hold on;
                title("Pupil x,y change (eye centered)")
                plot(ts_mins(2:end),dl_eye_nolicks(:,1));
                plot(ts_mins(2:end),-dl_eye_nolicks(:,2));
                plot(rmmissing(saccade_imu_table.left_ts+1),dl_eye_nolicks(rmmissing(saccade_imu_table.left_idx),1),'k.')
                plot(rmmissing(saccade_imu_table.right_ts+1),-dl_eye_nolicks(rmmissing(saccade_imu_table.right_idx),2),'k.')
                xlim([0 max(ts_mins)])
        
                subplot(5,1,4); hold on;
                title("Pupil x,y change (raw pupil)")
                plot(ts_mins(2:end),dl_camera_nolicks(:,1));
                plot(ts_mins(2:end),-dl_camera_nolicks(:,2));
                plot(rmmissing(saccade_imu_table.left_ts+1),dl_camera_nolicks(rmmissing(saccade_imu_table.left_idx),1),'k.')
                plot(rmmissing(saccade_imu_table.right_ts+1),-dl_camera_nolicks(rmmissing(saccade_imu_table.right_idx),2),'k.')
                xlabel('Time (min)');
                xlim([0 max(ts_mins)])
        
                subplot(5,1,5); hold on;
                title("Lick sensor")
                plot(ts_mins,licking);
                xlabel('Time (min)');
                xlim([0 max(ts_mins)])
                ylim([0 1]);
            
                savefile = strcat(summary_plots_save_directory,filename,'_sumplot1a_x&y&saccades_licksremoved.png');
                saveas(fig,savefile)
            end
        
            %%%make saccade summary plot 2 - polar histogram of all saccade directions (for both eyes side by side)
            fig = figure('position',[600,500,800,450]);
            subplot(1,2,1);
            sp1=gca;
            pos1 = sp1.Position;
            delete(sp1);
            pax=polaraxes('Position',pos1);
            polarhistogram(pax, deg2rad(rmmissing(saccade_imu_table.right_dir)),pi/32:pi/16:pi*65/32);
            title("Right eye")
            pax.ThetaZeroLocation='top';
            pax.ThetaDir='clockwise';
            pax.RLim = [0, max([pax.Children.BinCounts 1])];
            pax.ThetaTick = 0:30:330;  % Default positions (in degrees)
            % Set tick labels to custom ones from -180 to 180
            custom_labels = {'0','30','60','90','120','150','180',...
                     '-150','-120','-90','-60','-30'};
            pax.ThetaTickLabel = custom_labels;
            ax1=gca;

            subplot(1,2,2);
            sp1=gca;
            pos1 = sp1.Position;
            delete(sp1);
            pax=polaraxes('Position',pos1);
            polarhistogram(pax, deg2rad(rmmissing(saccade_imu_table.left_dir)),pi/32:pi/16:pi*65/32);
            title("Left eye")
            pax.ThetaZeroLocation='top';
            pax.ThetaDir='clockwise';
            pax.RLim = [0, max([pax.Children.BinCounts 1])];
            pax.ThetaTick = 0:30:330;  % Default positions (in degrees)
            pax.ThetaTickLabel = custom_labels;
            ax2=gca;
            sgtitle(strcat('Saccade directions (',filename,')'),"Interpreter","none");
            ax1.RLim(2) = max(ax1.RLim(2),ax2.RLim(2));
            ax2.RLim(2) = max(ax1.RLim(2),ax2.RLim(2));
            text(-110,-40,"Angle of saccade (degrees)",'Units','pixels')
            savefile = strcat(summary_plots_save_directory,filename,'_sumplot2_histogram_polar.png');
            saveas(fig,savefile)


            %%%make saccade summary plot 3 - histogram of all saccade directions (for both eyes side by side)
            fig = figure('position',[600,500,800,450]);
            subplot(1,2,1);
            histogram(rmmissing(saccade_imu_table.right_dir),-180:10:180)
            ylim1=ylim;
            ax1=gca;
            title("Right eye")
            subplot(1,2,2);
            histogram(rmmissing(saccade_imu_table.left_dir),-180:10:180)
            title("Left eye")
            ax2=gca;
            ylim2=ylim;
            sgtitle(strcat('Saccade directions (',filename,')'),"Interpreter","none");
            ylim(ax1,[0,max(ylim1(2),ylim2(2))]);
            ylim(ax2,[0,max(ylim1(2),ylim2(2))]);
            text(-90,-30,"Angle of saccade (degrees)",'Units','pixels')
            savefile = strcat(summary_plots_save_directory,filename,'_sumplot3_direction_histogram.png');
            saveas(fig,savefile)
        
            %%%make saccade summary plot 4 - histogram of all saccade magnitudes (for both eyes side by side)
            fig = figure('position',[600,500,800,450]);
            subplot(1,2,1);
            histogram(rmmissing(saccade_imu_table.left_mag),50)
            ylim1=ylim;
            ax1=gca;
            title("Left eye")
            subplot(1,2,2);
            histogram(rmmissing(saccade_imu_table.left_mag),50)
            title("Right eye")
            ax2=gca;
            ylim2=ylim;
            sgtitle(strcat('Saccade magnitudes (',filename,')'),"Interpreter","none");
            ylim(ax1,[0,max(ylim1(2),ylim2(2))]);
            ylim(ax2,[0,max(ylim1(2),ylim2(2))]);
            text(-90,-30,"Magnitude of saccade (degrees)",'Units','pixels')
            savefile = strcat(summary_plots_save_directory,filename,'_sumplot4_magnitude_histogram.png');
            saveas(fig,savefile)
        
            %%%make saccade summary plot 5 - histogram of time between saccades (for both eyes side by side)
            fig = figure('position',[600,500,800,450]);
            subplot(2,2,1);
            time_elapsed = diff(rmmissing(saccade_imu_table.left_ts))*parameters.framerate;
            histogram(time_elapsed,25)
            ylim1=ylim;
            xlim1=xlim;
            ax1=gca;
            title("Left eye (all saccades)")
            subplot(2,2,2);
            time_elapsed = diff(rmmissing(saccade_imu_table.right_ts))*parameters.framerate;
            histogram(time_elapsed,25)
            title("Right eye (all saccades)")
            ylim2=ylim;
            xlim2=xlim;
            ax2=gca;
            subplot(2,2,3);
            time_elapsed = diff(rmmissing(saccade_imu_table.left_ts))*parameters.framerate;
            histogram(time_elapsed,0:0.1:10)
            ylim3=ylim;
            ax3=gca;
            title("Left eye (zoomed in)")
            subplot(2,2,4);
            time_elapsed = diff(rmmissing(saccade_imu_table.right_ts))*parameters.framerate;
            histogram(time_elapsed,0:0.1:10)
            ylim4=ylim;
            ax4=gca;
            title("Right eye (zoomed in)")
            sgtitle(strcat('Time between saccades (',filename,')'),"Interpreter","none");
            ylim(ax1,[0,max([ylim1(2) ylim2(2) ylim3(2) ylim4(2)])]);
            ylim(ax2,[0,max([ylim1(2) ylim2(2) ylim3(2) ylim4(2)])]);
            ylim(ax3,[0,max([ylim1(2) ylim2(2) ylim3(2) ylim4(2)])]);
            ylim(ax4,[0,max([ylim1(2) ylim2(2) ylim3(2) ylim4(2)])]);
            xlim(ax1,[0,max([xlim1(2) xlim2(2)])])
            xlim(ax2,[0,max([xlim1(2) xlim2(2)])])
            text(-90,-30,"Seconds",'Units','pixels')
            savefile = strcat(summary_plots_save_directory,filename,'_sumplot5_timebetween_histogram.png');
            saveas(fig,savefile)
        
            %%%make saccade summary plot 6 - correlation of magnitude and direction of saccades from both eyes
            fig = figure('position',[600,500,800,450]);
            subplot(1,2,1); hold on;
            scatter(saccade_imu_table.left_mag,saccade_imu_table.right_mag,50,'.')
            xlabel("Left eye")
            ylabel("Right eye")
            xlimit = xlim;
            ylimit = ylim;
            try
                xlim([0,max([ylimit(2) xlimit(2)])*1.1])
                ylim([0,max([ylimit(2) xlimit(2)])*1.1])
            catch
            end
            title("Saccade Magnitude (degrees)")

            subplot(1,2,2); hold on;
            scatter(saccade_imu_table.left_dir,saccade_imu_table.right_dir,50,'.')
            xlabel("Left eye")
            ylabel("Right eye")
            xlim([-180 180])
            ylim([-180 180])
            title("Saccade Direction")
            sgtitle(strcat('Corr. between saccade mag and dir from both eyes (',filename,')'),"Interpreter","none");
            savefile = strcat(summary_plots_save_directory,filename,'_sumplot6_mag&dir_correlation.png');
            saveas(fig,savefile)                
        
            %%%make saccade summary plot 7 - all eye x/y positions with saccades overlaid 
            fig=figure('position',[0         200        1800         800]); 
            subplot(1,2,1)
            scatter(smoothed_x(:,2),smoothed_y(:,2),'k.');hold on;
            try
                xlim([min(smoothed_x(:,2)),max(smoothed_x(:,2))]); ylim([min(smoothed_y(:,2)),max(smoothed_y(:,2))]);
            catch
            end
            title("Right eye")
            axis equal
            xlabel('Degrees x');
            ylabel('Degrees y');
            set(gca,'YDir','reverse');
            plot(smoothed_x(rmmissing(saccade_imu_table.right_idx)+1,2),smoothed_y(rmmissing(saccade_imu_table.right_idx)+1,2),'or','MarkerSize',20)      
            xlines = [smoothed_x(rmmissing(saccade_imu_table.right_idx),2),smoothed_x(rmmissing(saccade_imu_table.right_idx)+1,2)]';
            ylines = [smoothed_y(rmmissing(saccade_imu_table.right_idx),2),smoothed_y(rmmissing(saccade_imu_table.right_idx)+1,2)]';
            line(xlines,ylines,'LineWidth',3,'Color','g');

            subplot(1,2,2)
            scatter(smoothed_x(:,1),smoothed_y(:,1),'k.');hold on;
            try
                xlim([min(smoothed_x(:,1)),max(smoothed_x(:,1))]); ylim([min(smoothed_y(:,1)),max(smoothed_y(:,1))]);
            catch
            
            end
            title("Left eye")
            axis equal
            xlabel('Degrees x');
            ylabel('Degrees y');
            set(gca,'YDir','reverse');
            plot(smoothed_x(rmmissing(saccade_imu_table.left_idx)+1,1),smoothed_y(rmmissing(saccade_imu_table.left_idx)+1,1),'or','MarkerSize',20)      
            xlines = [smoothed_x(rmmissing(saccade_imu_table.left_idx),1),smoothed_x(rmmissing(saccade_imu_table.left_idx)+1,1)]';
            ylines = [smoothed_y(rmmissing(saccade_imu_table.left_idx),1),smoothed_y(rmmissing(saccade_imu_table.left_idx)+1,1)]';
            line(xlines,ylines,'LineWidth',3,'Color','g');
            sgtitle(strcat('Eye position across session- green lines indicate saccades (',filename,')'),"Interpreter","none");
            savefile = strcat(summary_plots_save_directory,filename,'_sumplot7_saccadeplot.png');
            saveas(fig,savefile)

            % %%%make saccade summary plot 7a - all eye x/y positions with saccades overlaid - just 30 seconds with most saccades
            % most_saccades = 0;
            % most_saccades_idx = 1;
            % win = 30/60;
            % for i=0:win:floor(max(ts_mins))
            %     count_saccades = sum(ix>=i*parameters.framerate*60&ix<(i+win)*parameters.framerate*60);
            %     if count_saccades > most_saccades
            %         most_saccades_idx = i;
            %         most_saccades = count_saccades;
            %     end
            % end
            % 
            % most_saccades_frames = most_saccades_idx*parameters.framerate*60:(most_saccades_idx+win)*parameters.framerate*60;
            % most_saccades_frames = most_saccades_frames(most_saccades_frames<length(smoothed_x)&most_saccades_frames>0);
            % most_saccades_idxs_l = saccade_imu_table.left_idx(ismember(saccade_imu_table.left_idx,most_saccades_frames));
            % most_saccades_idxs_r = saccade_imu_table.right_idx(ismember(saccade_imu_table.right_idx,most_saccades_frames));
            % 
            % fig=figure('position',[0         200        1800         800]); 
            % subplot(1,2,1)
            % if ~isempty(most_saccades_idxs_r)
            %     scatter(smoothed_x(most_saccades_frames,2),smoothed_y(most_saccades_frames,2),'k.');hold on;
            %     try
            %         xlim([min(smoothed_x(:,2)),max(smoothed_x(:,2))]); ylim([min(smoothed_y(:,2)),max(smoothed_y(:,2))]);
            %     catch
            %     end
            %     title("Right eye")
            %     axis equal
            %     xlabel('Degrees x');
            %     ylabel('Degrees y');
            %     set(gca,'YDir','reverse');
            %     plot(smoothed_x(rmmissing(most_saccades_idxs_r)+1,2),smoothed_y(rmmissing(most_saccades_idxs_r)+1,2),'or','MarkerSize',20)      
            %     xlines = [smoothed_x(rmmissing(most_saccades_idxs_r),2),smoothed_x(rmmissing(most_saccades_idxs_r)+1,2)]';
            %     ylines = [smoothed_y(rmmissing(most_saccades_idxs_r),2),smoothed_y(rmmissing(most_saccades_idxs_r)+1,2)]';
            %     line(xlines,ylines,'LineWidth',3,'Color','g');
            % end
            % subplot(1,2,2)
            % if ~isempty(most_saccades_idxs_l)
            %     scatter(smoothed_x(most_saccades_frames,1),smoothed_y(most_saccades_frames,1),'k.');hold on;
            %     try
            %         xlim([min(smoothed_x(:,1)),max(smoothed_x(:,1))]); ylim([min(smoothed_y(:,1)),max(smoothed_y(:,1))]);
            %     catch
            %     end
            %     title("Left eye")
            %     axis equal
            %     xlabel('Degrees x');
            %     ylabel('Degrees y');
            %     set(gca,'YDir','reverse');
            %     plot(smoothed_x(rmmissing(most_saccades_idxs_l)+1,1),smoothed_y(rmmissing(most_saccades_idxs_l)+1,1),'or','MarkerSize',20)      
            %     xlines = [smoothed_x(rmmissing(most_saccades_idxs_l),1),smoothed_x(rmmissing(most_saccades_idxs_l)+1,1)]';
            %     ylines = [smoothed_y(rmmissing(most_saccades_idxs_l),1),smoothed_y(rmmissing(most_saccades_idxs_l)+1,1)]';
            %     line(xlines,ylines,'LineWidth',3,'Color','g');
            % end
            % 
            % sgtitle(strcat('Eye position across session- green lines indicate saccades - just 30 seconds with most saccades  (',filename,')'),"Interpreter","none");
            % savefile = strcat(summary_plots_save_directory,filename,'_sumplot7a_saccadeplot_30seconds.png');
            % saveas(fig,savefile)

        end

    
        %% make IMU summary plots - plots summarizing IMU over time and around all detected saccades
        if plotting_params.make_imu_summary_plots
            if ~isfolder(imu_summary_save_directory)
                mkdir(imu_summary_save_directory)
            end

            %%%make imu summary plot 1 - eye x/y position, saccades, pupil size, raw IMU signals, and licking over time
            fig=figure('position',[0         0        1800         1000]); 
            sgtitle(strcat('Pupil, saccades and IMU across time (',filename,')'),"Interpreter","none");

            subplot(6,1,1);hold on;
            title("Pupil x")
            plot(ts_mins,smoothed_x);
            plot(rmmissing(saccade_imu_table.left_ts),smoothed_x(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),smoothed_x(rmmissing(saccade_imu_table.right_idx),2),'k.')
            ylabel('Degrees');
            xlim([0 max(ts_mins)])
    
            subplot(6,1,2); hold on;
            title("Pupil y")
            plot(ts_mins,smoothed_y);
            plot(rmmissing(saccade_imu_table.left_ts),smoothed_y(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),smoothed_y(rmmissing(saccade_imu_table.right_idx),2),'k.')
            ylabel('Degrees');
            xlim([0 max(ts_mins)])
    
            subplot(6,1,3); hold on;
            title("Pupil size")
            plot(ts_mins,pupil_size);
            ylabel('Degrees');
            xlim([0 max(ts_mins)])

            subplot(6,1,4); hold on;
            title("IMU accelerometer")
            offset = mean(std(imu(:,1:3))*10);
            plot(ts_mins,imu(:,1)+offset);
            plot(ts_mins,imu(:,2));
            plot(ts_mins,imu(:,3)-offset);
            legend("accel x","accel y","accel z")
            ylabel('Gs');
            xlim([0 max(ts_mins)])
    
            subplot(6,1,5); hold on;
            title("IMU gyroscope")
            offset = mean(std(imu(:,4:6))*10);
            plot(ts_mins,imu(:,4)+offset);
            plot(ts_mins,imu(:,5));
            plot(ts_mins,imu(:,6)-offset);
            legend("gyro x","gyro y","gyro z")
            xlabel('Time (min)');
            ylabel('DPS');
            xlim([0 max(ts_mins)])

            subplot(6,1,6); hold on;
            title("Lick sensor")
            plot(ts_mins,licking);
            xlabel('Time (min)');
            xlim([0 max(ts_mins)])
            ylim([0 1])

            savefile = strcat(imu_summary_save_directory,filename,'_sumplot1_IMUsummary.png');
            saveas(fig,savefile)


            %%%make imu summary plot 2 - just eye x position and accelerometer over time
            fig=figure('position',[0         0        1800         1000]); 
            sgtitle(strcat('Eye x pos. and accelerometer across time (',filename,')'),"Interpreter","none");

            subplot(2,1,1);hold on;
            title("Pupil x")
            plot(ts_mins,smoothed_x);
            plot(rmmissing(saccade_imu_table.left_ts),smoothed_x(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),smoothed_x(rmmissing(saccade_imu_table.right_idx),2),'k.')
            ylabel('Degrees');
            xlim([0 max(ts_mins)])
    
            subplot(2,1,2); hold on;
            title("IMU accelerometer")
            offset = mean(std(imu(:,1:3))*10);
            plot(ts_mins,imu(:,1)+offset);
            plot(ts_mins,imu(:,2));
            plot(ts_mins,imu(:,3)-offset);
            ylabel('Gs');
            xlim([0 max(ts_mins)])
            xline(ts_mins(ix));

            savefile = strcat(imu_summary_save_directory,filename,'_sumplot2_IMUsummary_justeyexandaccel.png');
            saveas(fig,savefile)

            %%%make imu summary plot 3 - just eye x position and accelerometer over time (30 seconds zoomed in section)
            fig=figure('position',[0         0        1800         1000]); 
            sgtitle(strcat('Eye x pos. and accelerometer across time - 30 seconds with most saccades zoomed in (',filename,')'),"Interpreter","none");

            most_saccades = 0;
            most_saccades_idx = 1;
            for i=0:0.5:floor(max(ts_mins))
                count_saccades = sum(ix>=i*parameters.framerate*60&ix<(i+0.5)*parameters.framerate*60);
                if count_saccades > most_saccades
                    most_saccades_idx = i;
                    most_saccades = count_saccades;
                end
            end

            subplot(2,1,1);hold on;
            title("Pupil x")
            plot(ts_mins,smoothed_x);
            plot(rmmissing(saccade_imu_table.left_ts),smoothed_x(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),smoothed_x(rmmissing(saccade_imu_table.right_idx),2),'k.')
            ylabel('Degrees');
            xlim([most_saccades_idx most_saccades_idx+0.5])
    
            subplot(2,1,2); hold on;
            title("IMU accelerometer")
            offset = mean(std(imu(:,1:3))*10);
            plot(ts_mins,imu(:,1)+offset);
            plot(ts_mins,imu(:,2));
            plot(ts_mins,imu(:,3)-offset);
            ylabel('Gs');
            xline(ts_mins(ix));
            xlim([most_saccades_idx most_saccades_idx+0.5])

            savefile = strcat(imu_summary_save_directory,filename,'_sumplot3_IMUsummary_justeyexandaccel_zoomedin.png');
            saveas(fig,savefile)


            %%%make imu summary plot 1a - eye x/y position, saccades, pupil size, processed IMU signals, and licking over time
            fig=figure('position',[0         0        1800         1000]); 
            sgtitle(strcat('Pupil, saccades and processed IMU across time (',filename,')'),"Interpreter","none");

            subplot(6,1,1);hold on;
            title("Pupil x")
            plot(ts_mins,smoothed_x);
            plot(rmmissing(saccade_imu_table.left_ts),smoothed_x(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),smoothed_x(rmmissing(saccade_imu_table.right_idx),2),'k.')
            ylabel('Degrees');
            xlim([0 max(ts_mins)])
    
            subplot(6,1,2); hold on;
            title("Pupil y")
            plot(ts_mins,smoothed_y);
            plot(rmmissing(saccade_imu_table.left_ts),smoothed_y(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),smoothed_y(rmmissing(saccade_imu_table.right_idx),2),'k.')
            ylabel('Degrees');
            xlim([0 max(ts_mins)])
    
            subplot(6,1,3); hold on;
            title("Pupil size")
            plot(ts_mins,pupil_size);
            ylabel('Pixels^2');
            xlim([0 max(ts_mins)])

            subplot(6,1,4); hold on;
            title("IMU accelerometer")
            offset = mean(std(processed_imu(:,1:3))*10);
            plot(ts_mins,processed_imu(:,1)+offset);
            plot(ts_mins,processed_imu(:,2));
            plot(ts_mins,processed_imu(:,3)-offset);
            legend("processed x","processed y","processed z")
            xlim([0 max(ts_mins)])

            subplot(6,1,6); hold on;
            title("Lick sensor")
            plot(ts_mins,licking);
            xlabel('Time (min)');
            xlim([0 max(ts_mins)])
            ylim([0 1])

            savefile = strcat(imu_summary_save_directory,filename,'_sumplot1a_IMUsummary_processedIMU.png');
            saveas(fig,savefile)


            %%%make imu summary plot 2a - just eye x position and processed IMU over time
            fig=figure('position',[0         0        1800         1000]); 
            sgtitle(strcat('Eye x pos. and processed IMU across time (',filename,')'),"Interpreter","none");

            subplot(2,1,1);hold on;
            title("Pupil x")
            plot(ts_mins,smoothed_x);
            plot(rmmissing(saccade_imu_table.left_ts),smoothed_x(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),smoothed_x(rmmissing(saccade_imu_table.right_idx),2),'k.')
            ylabel('Degrees');
            xlim([0 max(ts_mins)])
    
            subplot(2,1,2); hold on;
            title("Processed IMU")
            offset = mean(std(processed_imu(:,1:3))*10);
            plot(ts_mins,processed_imu(:,1)+offset);
            plot(ts_mins,processed_imu(:,2));
            plot(ts_mins,processed_imu(:,3)-offset);
            xlim([0 max(ts_mins)])
            xline(ts_mins(ix));

            savefile = strcat(imu_summary_save_directory,filename,'_sumplot2a_IMUsummary_justeyexandaccel_processedIMU.png');
            saveas(fig,savefile)

            %%%make imu summary plot 3a - just eye x position and accelerometer over time (30 seconds zoomed in section)
            fig=figure('position',[0         0        1800         1000]); 
            sgtitle(strcat('Eye x pos. and processed IMU across time - 30 seconds with most saccades zoomed in (',filename,')'),"Interpreter","none");

            most_saccades = 0;
            most_saccades_idx = 1;
            for i=0:0.5:floor(max(ts_mins))
                count_saccades = sum(ix>=i*parameters.framerate*60&ix<(i+0.5)*parameters.framerate*60);
                if count_saccades > most_saccades
                    most_saccades_idx = i;
                    most_saccades = count_saccades;
                end
            end

            subplot(2,1,1);hold on;
            title("Pupil x")
            plot(ts_mins,smoothed_x);
            plot(rmmissing(saccade_imu_table.left_ts),smoothed_x(rmmissing(saccade_imu_table.left_idx),1),'k.')
            plot(rmmissing(saccade_imu_table.right_ts),smoothed_x(rmmissing(saccade_imu_table.right_idx),2),'k.')
            ylabel('Degrees');
            xlim([most_saccades_idx most_saccades_idx+0.5])
    
            subplot(2,1,2); hold on;
            title("Processed IMU")
            offset = mean(std(processed_imu(:,1:3))*10);
            plot(ts_mins,processed_imu(:,1)+offset);
            plot(ts_mins,processed_imu(:,2));
            plot(ts_mins,processed_imu(:,3)-offset);
            ylabel('Gs');
            xline(ts_mins(ix));
            xlim([most_saccades_idx most_saccades_idx+0.5])

            savefile = strcat(imu_summary_save_directory,filename,'_sumplot3a_IMUsummary_justeyexandaccel_processedIMU_zoomedin.png');
            saveas(fig,savefile)
        end

        if plotting_params.make_full_session_zoomedin_plots
            if ~isfolder(full_session_zoomedin_plots_save_directory)
                mkdir(full_session_zoomedin_plots_save_directory)
            end
            for i=0:1:floor(max(ts_mins))
                fig=figure('position',[0         0        1800         1000]); 
                sgtitle(strcat('Eye x pos. and processed IMU across time - 1 minute zoomed in (',filename,')'),"Interpreter","none");
    
                ymax = max(abs([smoothed_x smoothed_y]),[],'all');

                subplot(4,1,1);hold on;
                title("Pupil x")
                plot(ts_mins,smoothed_x);
                plot(rmmissing(saccade_imu_table.left_ts),smoothed_x(rmmissing(saccade_imu_table.left_idx),1),'k.')
                plot(rmmissing(saccade_imu_table.right_ts),smoothed_x(rmmissing(saccade_imu_table.right_idx),2),'k.')
                ylabel('Degrees');
                xlim([i i+1])
                ylim([-ymax ymax])

                subplot(4,1,2);hold on;
                title("Pupil y")
                plot(ts_mins,smoothed_y);
                plot(rmmissing(saccade_imu_table.left_ts),smoothed_y(rmmissing(saccade_imu_table.left_idx),1),'k.')
                plot(rmmissing(saccade_imu_table.right_ts),smoothed_y(rmmissing(saccade_imu_table.right_idx),2),'k.')
                ylabel('Degrees');
                xlim([i i+1])
                ylim([-ymax ymax])

                subplot(4,1,3); hold on;
                title("Pupil x,y change (raw pupil)")
                plot(ts_mins(2:end),dl_camera(:,1));
                plot(ts_mins(2:end),-dl_camera(:,2));
                plot(ts_mins(rmmissing(saccade_imu_table.left_idx+1)),dl_camera(rmmissing(saccade_imu_table.left_idx),1),'k.')
                plot(ts_mins(rmmissing(saccade_imu_table.right_idx+1)),-dl_camera(rmmissing(saccade_imu_table.right_idx),2),'k.')
                xlabel('Time (min)');
                xlim([i i+1])
        
                subplot(4,1,4); hold on;
                title("Processed IMU")
                offset = mean(std(processed_imu(:,1:3))*10);
                plot(ts_mins,processed_imu(:,1)+offset);
                plot(ts_mins,processed_imu(:,2));
                plot(ts_mins,processed_imu(:,3)-offset);
                ylabel('Gs');
                xline(ts_mins(ix));
                xlim([i i+1])
    
                savefile = strcat(full_session_zoomedin_plots_save_directory,filename,'_sumplot3a_IMUsummary_justeyexandaccel_processedIMU_zoomedin_min',num2str(i),'.png');
                saveas(fig,savefile)
            end
        end
           
        %% make imu videos to show IMU and saccade
        if plotting_params.make_imu_videos
            if ~isfolder(imu_video_save_directory)
                mkdir(imu_video_save_directory)
            end
            if ~isfolder(imu_video_save_directory_licks) && parameters.separate_licking
                mkdir(imu_video_save_directory_licks)
            end

            if length(ix)>plotting_params.num_saccades_to_plot & plotting_params.plot_random_subset
                saccades_to_plot = sort(randsample(length(ix),plotting_params.num_saccades_to_plot))';
            else
                saccades_to_plot = 1:length(ix);
            end

            % saccades_to_plot = potential_saccades_to_combine;

            % saccades_to_plot = 1:length(ix);
            % saccades_to_plot = saccades_to_plot(saccade_imu_table.head_movement==1);

            for j = saccades_to_plot
                % try
                    sac = ix(j);
                    plot_window_size = round(parameters.framerate*2);
                    plot_window = sac-plot_window_size:sac+plot_window_size;
                    plot_ts = (plot_window-sac)/parameters.framerate;
                    animate_window_size = round(parameters.framerate);
                    animate_window = sac-animate_window_size:sac+animate_window_size;
                    animate_ts= (-animate_window_size:animate_window_size)/parameters.framerate;
    
                    if saccade_imu_table.lick_sum(j)>0 && parameters.separate_licking
                        saccadeVideo=VideoWriter(strjoin([imu_video_save_directory_licks,filename,'_saccade',num2str(j)],''),'MPEG-4');
                    else
                        saccadeVideo=VideoWriter(strjoin([imu_video_save_directory,filename,'_saccade',num2str(j)],''),'MPEG-4');
                    end
                    saccadeVideo.FrameRate = 10;
                    open(saccadeVideo);
    
                    if saccade_imu_table.left_mag(j) > saccade_imu_table.right_mag(j) || isnan(saccade_imu_table.right_mag(j))
                        frames_eye = read(eye_left_vidObj,[sac-animate_window_size sac+animate_window_size]); %TEMPORARY TESTING BC THE CAMERA FRAMES ARE 1 OFF FROM EACHOTHER (SHOULD CHECK THIS IS HOW IT WORKS IN BONSAI)
                        eye_vidObj = eye_left_vidObj;
                        eye_idx = 1;
                        other_eye_idx = 2;
                        % frames_head = read(head_right_vidObj,[sac-animate_window_size sac+animate_window_size]); %TEMPORARY TESTING BC THE CAMERA FRAMES ARE 1 OFF FROM EACHOTHER (SHOULD CHECK THIS IS HOW IT WORKS IN BONSAI)
                        % head_vidObj = head_right_vidObj;
                        frames_head = read(head_left_vidObj,[sac-animate_window_size sac+animate_window_size]); %TEMPORARY TESTING BC THE CAMERA FRAMES ARE 1 OFF FROM EACHOTHER (SHOULD CHECK THIS IS HOW IT WORKS IN BONSAI)
                        head_vidObj = head_left_vidObj;
                    else
                        frames_eye = read(eye_right_vidObj,[sac-animate_window_size+1 sac+animate_window_size+1]);
                        frames_eye = flip(frames_eye,2);
                        eye_vidObj = eye_right_vidObj;
                        eye_idx = 2;
                        other_eye_idx = 1;
                        % frames_head = read(head_left_vidObj,[sac-animate_window_size sac+animate_window_size]); %TEMPORARY TESTING BC THE CAMERA FRAMES ARE 1 OFF FROM EACHOTHER (SHOULD CHECK THIS IS HOW IT WORKS IN BONSAI)
                        % head_vidObj = head_left_vidObj;
                        frames_head = read(head_right_vidObj,[sac-animate_window_size sac+animate_window_size]); %TEMPORARY TESTING BC THE CAMERA FRAMES ARE 1 OFF FROM EACHOTHER (SHOULD CHECK THIS IS HOW IT WORKS IN BONSAI)
                        head_vidObj = head_right_vidObj;
                    end
    
                    k=1;
    
                    figure('position',[0 0 1920 1080]);
    
                    % set up lick plotting coordinates
                    lick_diff = diff(licking(sac-plot_window_size+1:sac+plot_window_size))';
                    licks_on = find(lick_diff==1);
                    licks_off = find(lick_diff==-1);
                    if ~isempty(licks_on) || ~isempty(licks_off)
                        if isempty(licks_on) && isscalar(licks_off)
                            licks_on = 1;
                        end
                        if isempty(licks_off) && isscalar(licks_on)
                            licks_off = length(plot_ts);
                        end
                        if licks_on(1)>licks_off(1)
                            licks_on = [1 licks_on];
                        end
                        if licks_on(end)>licks_off(end)
                            licks_off = [licks_off length(plot_ts)];
                        end
                    end
                    lick_plot_coords = [licks_on;licks_off];
    
                    %set up all plots
                    subplot(2,3,1); hold on; %eye video
                    xlim([0 eye_vidObj.Width]); ylim([0 eye_vidObj.Height]);    
                    if saccade_imu_table.left_mag(j) > saccade_imu_table.right_mag(j) || isnan(saccade_imu_table.right_mag(j))
                        title("Left eye")
                    else
                        title("Right eye")
                    end
                        
                    subplot(2,3,4); hold on; %head video of opposite side of face
                    if saccade_imu_table.left_mag(j) > saccade_imu_table.right_mag(j) || isnan(saccade_imu_table.right_mag(j))
                        title("Left head")
                    else
                        title("Right head")
                    end
    
                    subplot(2,3,2); hold on; %eye x/y plot
                    xlim([min(smoothed_x,[],'all'),max(smoothed_x,[],'all')]); ylim([min(smoothed_y,[],'all'),max(smoothed_y,[],'all')]);
                    axis equal
                    set(gca,'YDir','reverse');
                    xlabel("Degrees x")
                    ylabel("Degrees y")
                    title("Eye x,y position")
    
                    subplot(2,3,3); hold on; %eye centered
                    p1 = plot(plot_ts,smoothed_x(plot_window,eye_idx)-mean(smoothed_x(plot_window,eye_idx),"omitnan"));
                    p2 = plot(plot_ts,smoothed_y(plot_window,eye_idx)-mean(smoothed_y(plot_window,eye_idx),"omitnan"));
                    p3 = plot(plot_ts,dl(plot_window,eye_idx));

                    xlim([plot_ts(1) plot_ts(end)]);
                    % ylim([-10 10])
                    bar1 = xline(0,'r');
                    xline(0,'--k')
                    xlabel("Time since saccade (sec)")
                    ylabel("Degrees")
                    title("Pupil position")
                    legend([p1 p2 p3],{"pupil x","pupil y","pupil dist. moved"})
                    yl = ylim;
                    y = yl(1);
                    h = yl(2) - yl(1);
                    for i=1:size(lick_plot_coords,2)
                        x = plot_ts(lick_plot_coords(1,i));
                        w = plot_ts(lick_plot_coords(2,i)) - plot_ts(lick_plot_coords(1,i));
                        if w<0
                            continue;
                        end
                        rectangle('position', [x y w h],"FaceColor",'b',"FaceAlpha",0.2,'edgecolor', 'none');
                    end
    
                    subplot(2,3,5); hold on; %IMU accel x/y/z
                    offset = mean(std(imu(:,1:3))*10);
                    p4 = plot(plot_ts,imu(plot_window,1)+offset);
                    p5 = plot(plot_ts,imu(plot_window,2));
                    p6 = plot(plot_ts,imu(plot_window,3)-offset);
                    p4a = plot(plot_ts,processed_imu(plot_window,1)+offset);
                    p5a = plot(plot_ts,processed_imu(plot_window,2));
                    p6a = plot(plot_ts,processed_imu(plot_window,3)-offset);
                    xlim([plot_ts(1) plot_ts(end)]);
                    yl = ylim;
                    bar7 = xline(0,'r');
                    xline(0,'--k')
                    xlabel("Time since saccade (sec)")
                    ylabel("Gs (shifted for plotting)")
                    title("IMU - accelerometer")
                    legend([p4 p5 p6],{"accel x","accel y","accel z"})
                    y = yl(1);
                    h = yl(2) - yl(1);
                    for i=1:size(lick_plot_coords,2)
                        x = plot_ts(lick_plot_coords(1,i));
                        w = plot_ts(lick_plot_coords(2,i)) - plot_ts(lick_plot_coords(1,i));
                        if w<0
                            continue;
                        end
                        rectangle('position', [x y w h],"FaceColor",'b',"FaceAlpha",0.2,'edgecolor', 'none');
                    end
    
                    subplot(2,3,6); hold on; %IMU gyro x/y/z
                    offset = mean(std(imu(:,4:6))*10);
                    p7 = plot(plot_ts,imu(plot_window,4)+offset);
                    p8 = plot(plot_ts,imu(plot_window,5));
                    p9 = plot(plot_ts,imu(plot_window,6)-offset);
                    xlim([plot_ts(1) plot_ts(end)]);
                    yl = ylim;
                    bar11 = xline(0,'r');
                    xline(0,'--k')
                    xlabel("Time since saccade (sec)")
                    ylabel("DPS (shifted for plotting)")
                    title("IMU - gyroscope")
                    legend([p7 p8 p9],{"gyro x","gyro y","gyro z"})
                    y = yl(1);
                    h = yl(2) - yl(1);
                    for i=1:size(lick_plot_coords,2)
                        x = plot_ts(lick_plot_coords(1,i));
                        w = plot_ts(lick_plot_coords(2,i)) - plot_ts(lick_plot_coords(1,i));
                        if w<0
                            continue;
                        end
                        rectangle('position', [x y w h],"FaceColor",'b',"FaceAlpha",0.2,'edgecolor', 'none');
                    end
    
    
                    sgtitle([filename,' (Sac. mag. (L): ',num2str(saccade_imu_table.left_mag(j),'%6.2f'),', (R): ',num2str(saccade_imu_table.right_mag(j),'%6.2f'),' Frame num: ',num2str(sac),')'],'Interpreter','none');
    
    
                    for i = animate_window
    
                        subplot(2,3,1); hold on; %eye video
                        frame = frames_eye(:,:,:,k);
                        imshow(frame); axis on; hold on;
                        % xlabel('Degrees x'); ylabel('Degrees y');
                        current_axis = gca;
                        if ~anynan(x_pupil(i,:,eye_idx)) && ~anynan(y_pupil(i,:,eye_idx))
                            ellipse_t = fit_ellipse(x_pupil(i,:,eye_idx)',y_pupil(i,:,eye_idx)',current_axis);
                        end        
                        plot(center_x_rawcoords(i-1,eye_idx),center_y_rawcoords(i-1,eye_idx),'r.','MarkerSize',20);
                        plot(center_x_rawcoords(i,eye_idx),center_y_rawcoords(i,eye_idx),'g.','MarkerSize',20);
                        if ismember(i,saccade_imu_table.saccade_frame+1)
                            title('Saccade!');
                        else
                            if saccade_imu_table.left_mag(j) > saccade_imu_table.right_mag(j) || isnan(saccade_imu_table.right_mag(j))
                                title("Left eye")
                            else
                                title("Right eye")
                            end                        
                        end
    
                        subplot(2,3,4); hold on; %head video
                        frame = frames_head(:,:,:,k);
                        imshow(frame); axis on; hold on;
                        xlim([0 head_vidObj.Width]); ylim([0 head_vidObj.Height]); 
    
                        subplot(2,3,2); hold on; %eye x/y plot
                        idx=sac-animate_window_size:i;
                        scatter(smoothed_x(idx(idx>0),eye_idx),smoothed_y(idx(idx>0),eye_idx),10,'k',"filled");
                        plot(smoothed_x(i,eye_idx),smoothed_y(i,eye_idx),'r.','MarkerSize',9);
                        % if ismember(i,saccade_imu_table.saccade_frame+1)
                        %     plot(smoothed_x(i,eye_idx),smoothed_y(i,eye_idx),'or','MarkerSize',20)      
                        %     line([smoothed_x(i,eye_idx) smoothed_x(i-1,eye_idx)],[smoothed_y(i,eye_idx) smoothed_y(i-1,eye_idx)],'LineWidth',5,'Color','g');
                        % end
                        % if ismember(i,ix_end_cell{eye_idx})
                        %     plot(smoothed_x(i,eye_idx),smoothed_y(i,eye_idx),'or','MarkerSize',20)      
                        %     line([smoothed_x(i,eye_idx) smoothed_x(ix_start_cell{eye_idx}(j),eye_idx)],[smoothed_y(i,eye_idx) smoothed_y(ix_start_cell{eye_idx}(j),eye_idx)],'LineWidth',5,'Color','g');
                        % end
                        if ismember(i,ix_start_cell{eye_idx})
                            [a,b]=ismember(i,ix_start_cell{eye_idx});
                            plot(smoothed_x(ix_end_cell{eye_idx}(b),eye_idx),smoothed_y(ix_end_cell{eye_idx}(b),eye_idx),'ok','MarkerSize',20)      
                            line([smoothed_x(ix_start_cell{eye_idx}(b),eye_idx) smoothed_x(ix_end_cell{eye_idx}(b),eye_idx)],[smoothed_y(ix_start_cell{eye_idx}(b),eye_idx) smoothed_y(ix_end_cell{eye_idx}(b),eye_idx)],'LineWidth',5,'Color','b');
                        end
    
                        idx=sac-animate_window_size:i;
                        scatter(smoothed_x(idx(idx>0),other_eye_idx),smoothed_y(idx(idx>0),other_eye_idx),10,[0.5 0.5 0.5],"filled");
                        plot(smoothed_x(i,other_eye_idx),smoothed_y(i,other_eye_idx),'r.','MarkerSize',9);
                        % if ismember(i,saccade_imu_table.saccade_frame+1)
                        %     plot(smoothed_x(i,other_eye_idx),smoothed_y(i,other_eye_idx),'or','MarkerSize',20)      
                        %     line([smoothed_x(i,other_eye_idx) smoothed_x(i-1,other_eye_idx)],[smoothed_y(i,other_eye_idx) smoothed_y(i-1,other_eye_idx)],'LineWidth',5,'Color','g');
                        % end
                        if ismember(i,ix_start_cell{other_eye_idx})
                            [a,b]=ismember(i,ix_start_cell{other_eye_idx});
                            plot(smoothed_x(ix_end_cell{other_eye_idx}(b),other_eye_idx),smoothed_y(ix_end_cell{other_eye_idx}(b),other_eye_idx),'ok','MarkerSize',20)      
                            line([smoothed_x(ix_start_cell{other_eye_idx}(b),other_eye_idx) smoothed_x(ix_end_cell{other_eye_idx}(b),other_eye_idx)],[smoothed_y(ix_start_cell{other_eye_idx}(b),other_eye_idx) smoothed_y(ix_end_cell{other_eye_idx}(b),other_eye_idx)],'LineWidth',5,'Color','b');
                        end


                        subplot(2,3,3); hold on; %eye centered
                        set(bar1,"Visible","off")
                        bar1 = xline(animate_ts(k),'r');
                        legend([p1 p2 p3],{"pupil x","pupil y","pupil dist. moved"})
    
                        subplot(2,3,5); hold on; %IMU accel x
                        set(bar7,"Visible","off")
                        bar7 = xline(animate_ts(k),'r');
                        legend([p4 p5 p6],{"accel x","accel y","accel z"})
    
                        subplot(2,3,6); hold on; %IMU gyro x
                        set(bar11,"Visible","off")
                        bar11 = xline(animate_ts(k),'r');
                        legend([p7 p8 p9],{"gyro x","gyro y","gyro z"})
    
                        k=k+1;
    
                        frame = getframe(gcf);
                        writeVideo(saccadeVideo,frame);
                        % if sum(ismember(i-3:i+1,ix))>0
                        %     pause;
                        %     % pause(0.5)
                        % end
                    end
                    close(saccadeVideo)
                    close all
                % catch
                %     continue;
                % end
            end
        end

        %% make saccade videos to show both eyes
        if plotting_params.make_saccade_videos
            if ~isfolder(saccade_video_save_directory)
                mkdir(saccade_video_save_directory)
            end

            if length(ix)>plotting_params.num_saccades_to_plot & plotting_params.plot_random_subset
                saccades_to_plot = sort(randsample(length(ix),plotting_params.num_saccades_to_plot))';
            else
                saccades_to_plot = 1:length(ix);
            end

            for j = saccades_to_plot
                % try
                    sac = ix(j);
                    plot_window_size = round(parameters.framerate*2);
                    plot_window = sac-plot_window_size:sac+plot_window_size;
                    plot_ts = (plot_window-sac)/parameters.framerate;
                    animate_window_size = round(parameters.framerate/2);
                    animate_window = sac-animate_window_size:sac+animate_window_size;
                    animate_ts= (-animate_window_size:animate_window_size)/parameters.framerate;
    
                    saccadeVideo=VideoWriter(strjoin([saccade_video_save_directory,filename,'_saccade',num2str(j)],''),'MPEG-4');
                    saccadeVideo.FrameRate = 10;
                    open(saccadeVideo);
    
                    frames_eye_left = read(eye_left_vidObj,[sac-animate_window_size sac+animate_window_size]); %TEMPORARY TESTING BC THE CAMERA FRAMES ARE 1 OFF FROM EACHOTHER (SHOULD CHECK THIS IS HOW IT WORKS IN BONSAI)
                    frames_eye_right = read(eye_right_vidObj,[sac-animate_window_size sac+animate_window_size]); %TEMPORARY TESTING BC THE CAMERA FRAMES ARE 1 OFF FROM EACHOTHER (SHOULD CHECK THIS IS HOW IT WORKS IN BONSAI)
                    frames_eye_right = flip(frames_eye_right,2);

                    k=1;
    
                    figure('position',[0 0 1920 1080]);
    
                    %set up all plots
                    subplot(2,3,2); hold on; %left eye video
                    xlim([0 eye_left_vidObj.Width]); ylim([0 eye_left_vidObj.Height]);    
                    title("Left eye")

                    subplot(2,3,5); hold on; %left eye x/y plot
                    xlim([min(smoothed_x,[],'all'),max(smoothed_x,[],'all')]); ylim([min(smoothed_y,[],'all'),max(smoothed_y,[],'all')]);
                    axis equal
                    set(gca,'YDir','reverse');
                    xlabel("Degrees x")
                    ylabel("Degrees y")
                    title(['L Dir = ', num2str(saccade_imu_table.left_dir(j),'%.1f')])

                    subplot(2,3,6); hold on; %left eye centered
                    p1 = plot(plot_ts,smoothed_x(plot_window,1)-mean(smoothed_x(plot_window,1),"omitnan"));
                    p2 = plot(plot_ts,smoothed_y(plot_window,1)-mean(smoothed_y(plot_window,1),"omitnan"));
                    p3 = plot(plot_ts,dl(plot_window,1));
                    xlim([plot_ts(1) plot_ts(end)]);
                    % ylim([-10 10])
                    bar1 = xline(0,'r');
                    xline(0,'--k')
                    yline(parameters.saccade_thresh)
                    xlabel("Time since saccade (sec)")
                    ylabel("Degrees")
                    title(['L Mag = ', num2str(saccade_imu_table.left_mag(j),'%.2f')])
                    legend([p1 p2 p3],{"pupil x","pupil y","pupil dist. moved"})


                    subplot(2,3,1); hold on; %right eye video
                    xlim([0 eye_right_vidObj.Width]); ylim([0 eye_right_vidObj.Height]);    
                    title("Right eye")
    
                    subplot(2,3,4); hold on; %right eye x/y plot
                    xlim([min(smoothed_x,[],'all'),max(smoothed_x,[],'all')]); ylim([min(smoothed_y,[],'all'),max(smoothed_y,[],'all')]);
                    axis equal
                    set(gca,'YDir','reverse');
                    xlabel("Degrees x")
                    ylabel("Degrees y")
                    title(['R Dir = ', num2str(saccade_imu_table.right_dir(j),'%.1f')])

                    subplot(2,3,3); hold on; %right eye centered
                    p4 = plot(plot_ts,smoothed_x(plot_window,2)-mean(smoothed_x(plot_window,2),"omitnan"));
                    p5 = plot(plot_ts,smoothed_y(plot_window,2)-mean(smoothed_y(plot_window,2),"omitnan"));
                    p6 = plot(plot_ts,dl(plot_window,2));
                    xlim([plot_ts(1) plot_ts(end)]);
                    % ylim([-10 10])
                    bar2 = xline(0,'r');
                    xline(0,'--k')
                    yline(parameters.saccade_thresh)
                    xlabel("Time since saccade (sec)")
                    ylabel("Degrees")
                    title(['R Mag = ', num2str(saccade_imu_table.right_mag(j),'%.2f')])
                    legend([p4 p5 p6],{"pupil x","pupil y","pupil dist. moved"})
   
                    sgtitle([filename,' (Frame num: ',num2str(sac),')'],'Interpreter','none');
    
    
                    for i = animate_window
                        subplot(2,3,2); hold on; %left eye video
                        frame = frames_eye_left(:,:,:,k);
                        imshow(frame); axis on; hold on;
                        current_axis = gca;
                        if ~anynan(x_pupil(i,:,1)) && ~anynan(y_pupil(i,:,1))
                            ellipse_t = fit_ellipse(x_pupil(i,:,1)',y_pupil(i,:,1)',current_axis);
                        end        
                        plot(center_x_rawcoords(i-1,1),center_y_rawcoords(i-1,1),'r.','MarkerSize',20);
                        plot(center_x_rawcoords(i,1),center_y_rawcoords(i,1),'g.','MarkerSize',20);
                        if ismember(i,saccade_imu_table.left_idx+1)
                            title('Saccade!');
                        else
                            title("Left eye")
                        end
    
                        subplot(2,3,1); hold on; %right eye video
                        frame = frames_eye_right(:,:,:,k);
                        imshow(frame); axis on; hold on;
                        current_axis = gca;
                        if ~anynan(x_pupil(i,:,2)) && ~anynan(y_pupil(i,:,2))
                            ellipse_t = fit_ellipse(x_pupil(i,:,2)',y_pupil(i,:,2)',current_axis);
                        end        
                        plot(center_x_rawcoords(i-1,2),center_y_rawcoords(i-1,2),'r.','MarkerSize',20);
                        plot(center_x_rawcoords(i,2),center_y_rawcoords(i,2),'g.','MarkerSize',20);
                        if ismember(i,saccade_imu_table.right_idx+1)
                            title('Saccade!');
                        else
                            title("Right eye")
                        end

                        subplot(2,3,5); hold on; %left eye x/y plot
                        idx=sac-animate_window_size:i;
                        scatter(smoothed_x(idx(idx>0),1),smoothed_y(idx(idx>0),1),10,'k',"filled");
                        plot(smoothed_x(i,1),smoothed_y(i,1),'r.','MarkerSize',9);
                        if ismember(i,ix_start_cell{1})
                            [a,b]=ismember(i,ix_start_cell{1});
                            plot(smoothed_x(ix_end_cell{1}(b),1),smoothed_y(ix_end_cell{1}(b),1),'ok','MarkerSize',20)      
                            line([smoothed_x(ix_start_cell{1}(b),1) smoothed_x(ix_end_cell{1}(b),1)],[smoothed_y(ix_start_cell{1}(b),1) smoothed_y(ix_end_cell{1}(b),1)],'LineWidth',5,'Color','b');
                        end
    
                        subplot(2,3,4); hold on; %right eye x/y plot
                        idx=sac-animate_window_size:i;
                        scatter(smoothed_x(idx(idx>0),2),smoothed_y(idx(idx>0),2),10,'k',"filled");
                        plot(smoothed_x(i,2),smoothed_y(i,2),'r.','MarkerSize',9);
                        if ismember(i,ix_start_cell{2})
                            [a,b]=ismember(i,ix_start_cell{2});
                            plot(smoothed_x(ix_end_cell{2}(b),2),smoothed_y(ix_end_cell{2}(b),2),'ok','MarkerSize',20)      
                            line([smoothed_x(ix_start_cell{2}(b),2) smoothed_x(ix_end_cell{2}(b),2)],[smoothed_y(ix_start_cell{2}(b),2) smoothed_y(ix_end_cell{2}(b),2)],'LineWidth',5,'Color','b');
                        end

                        subplot(2,3,6); hold on; %left eye centered
                        set(bar1,"Visible","off")
                        bar1 = xline(animate_ts(k),'r');
                        legend([p1 p2 p3],{"pupil x","pupil y","pupil dist. moved"})
        
                        subplot(2,3,3); hold on; %right eye centered
                        set(bar2,"Visible","off")
                        bar2 = xline(animate_ts(k),'r');
                        legend([p4 p5 p6],{"pupil x","pupil y","pupil dist. moved"})

                        k=k+1;
    
                        frame = getframe(gcf);
                        writeVideo(saccadeVideo,frame);
                        % if sum(ismember(i-3:i+1,ix))>0
                        %     pause;
                        %     % pause(0.5)
                        % end
                    end
                    close(saccadeVideo)
                    close all
                % catch
                %     continue;
                % end
            end


        end

        %% make summary plots per day


    end

    %% concatenate mass table of all saccade/imu information for all sessions in folder
    if exist('saccade_imu_population_table','var')
        save(strcat(saccade_imu_table_fname,'.mat'),'saccade_imu_population_table')
    end
    if height(dlc_likelihood_table)>0
        save(strcat(dlc_likelihood_table_fname,'.mat'),'dlc_likelihood_table')
        fig = figure;
        bar(dlc_likelihood_table.likelihoods)
        savefile = strcat(figures_save_directory,filename,'_dlc_likelihoods_by_day.png');
        saveas(fig,savefile)    
        fig = figure;
        bar(dlc_likelihood_table.likelihoods')
        savefile = strcat(figures_save_directory,filename,'_dlc_likelihoods_by_keypoint.png');
        saveas(fig,savefile)    

        fig = figure;
        subplot(2,2,1)
        bar(mean(dlc_likelihood_table.likelihoods(:,1:12)))
        ylim([0.8 1])
        title("Left eye keypoints")
        subplot(2,2,2)
        bar(mean(dlc_likelihood_table.likelihoods(:,13:24)))
        ylim([0.8 1])
        title("Right eye keypoints")
        subplot(2,2,3)
        bar(mean(dlc_likelihood_table.likelihoods(:,1:12),2))
        ylim([0.8 1])
        title("Left eye by day")
        subplot(2,2,4)
        bar(mean(dlc_likelihood_table.likelihoods(:,13:24),2))
        ylim([0.8 1])
        title("Right eye by day")
        savefile = strcat(figures_save_directory,filename,'_dlc_likelihoods_means.png');
        saveas(fig,savefile)   

        % fig = figure;
        % subplot(2,2,1)
        % bar(mean(dlc_likelihood_table.likelihoods(:,5:12)))
        % ylim([0.8 1])
        % title("Left eye keypoints")
        % subplot(2,2,2)
        % bar(mean(dlc_likelihood_table.likelihoods(:,17:24)))
        % ylim([0.8 1])
        % title("Right eye keypoints")
        % subplot(2,2,3)
        % bar(mean(dlc_likelihood_table.likelihoods(:,5:12),2))
        % ylim([0.8 1])
        % title("Left eye by day")
        % subplot(2,2,4)
        % bar(mean(dlc_likelihood_table.likelihoods(:,17:24),2))
        % ylim([0.8 1])
        % title("Right eye by day")

    end
end