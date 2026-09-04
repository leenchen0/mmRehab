import numpy as np
import struct
from signal_processing.radar_util import TI_RADAR_TYPE, CHIRP_CFG_DICT, REORDEING, ANT_PHASE_ROTATE

N_LANE          = 2
N_BYTE_UDP      = 1456 
N_UINT16_UDP    = int(N_BYTE_UDP/2)

def parse(file_name=None, adc_data = None, radar_config=None, debug=False):
    assert radar_config is not None, "radar_config should not be None"
    n_tx           = radar_config['n_tx']
    n_rx           = radar_config['n_rx']
    n_virtual_ant  = n_tx * n_rx
    n_sample       = radar_config['n_sample']
    n_loop         = radar_config['n_loop']
    n_uint16_frame = n_tx * n_rx * n_loop * n_sample * 2

    if file_name:
        adc_data = np.fromfile(file_name, dtype=np.int16)
    

    n_frame                 = adc_data.shape[0] // n_uint16_frame
    last_frame_start_index  = n_frame * n_uint16_frame
    adc_data                = adc_data[:last_frame_start_index]

    adc_data_complex = np.zeros(shape=(len(adc_data)//2), dtype=complex)
    adc_data_complex[0::2] = adc_data[0::4]+1j*adc_data[2::4]
    adc_data_complex[1::2] = adc_data[1::4]+1j*adc_data[3::4]

    frames = np.reshape(adc_data_complex, (n_frame, n_loop, n_virtual_ant, n_sample))
    if debug:
        print(f"adc_data_complex.shape: {adc_data_complex.shape}")
        print(f"frames.shape (n_frame, n_loop, n_virtual_ant, n_sample): {frames.shape}")

    return frames

def parse_with_timestamp(file_name='./data_collection/example/20240126_115711_086375.bin', radar_config=None, debug=False):
    assert radar_config is not None, "radar_config should not be None"
    n_tx           = radar_config['n_tx']
    n_rx           = radar_config['n_rx']
    n_virtual_ant  = n_tx * n_rx
    n_sample       = radar_config['n_sample']
    n_loop         = radar_config['n_loop']
    n_uint16_frame = n_tx * n_rx * n_loop * n_sample * 2

    data_list = []
    with open(file_name, 'rb') as file:
        while True:

            time_bytes = file.read(22)
            if not time_bytes:

                break
            time_str = time_bytes.decode('utf-8')


            packet_num_bytes = file.read(4)
            byte_count_bytes = file.read(6)
            packet_data_bytes = file.read(1456)
            packet_num = struct.unpack('<1l', packet_num_bytes)
            byte_count = struct.unpack('>Q', b'\x00\x00' + byte_count_bytes[::-1])
            packet_data = np.frombuffer(packet_data_bytes, dtype=np.int16)


            data_list.append({
                'time': time_str,
                'packet_num': packet_num,
                'byte_count': byte_count,
                'packet_data': packet_data
            })

    packet_nums = np.array([data["packet_num"][0] for data in data_list])
    if np.argmin(packet_nums) == 0:
        beg_i               = 0
        end_i               = len(data_list) - 1
    else:
        beg_i               = np.argmin(packet_nums)
        end_i               = len(data_list) - 1
    
    beg_packet_num      = data_list[beg_i]["packet_num"][0]
    end_packet_num      = data_list[end_i]["packet_num"][0]
    num_packet_need     = end_packet_num - beg_packet_num + 1
    num_packet_recieved = len(data_list) - beg_i

    udp_lost_ratio = 1 - (num_packet_recieved / num_packet_need)
    if udp_lost_ratio > 0.1:
        print(f"num_packet_need:{num_packet_need}, num_packet_recieved:{num_packet_recieved}, udp_lost_ratio:{udp_lost_ratio}")
        raise ValueError("Error: UDP packet loss rate exceeds 10%, program terminated.")
    
    # zero-padding
    adc_data = np.zeros((num_packet_need, N_UINT16_UDP))
    for i in range(beg_i, end_i + 1):
        packet_i            = data_list[i]["packet_num"][0] - beg_packet_num
        adc_data[packet_i]  = data_list[i]["packet_data"]

    if debug:
        print(f"Received {len(data_list)} packets")
        print(f"Need {num_packet_need} packets")
        print(f"[zero-padding] adc_data.shape: {adc_data.shape}")

    adc_data = adc_data.reshape(-1)


    beg_byte_count  = data_list[beg_i]["byte_count"][0]
    mod             = beg_byte_count%(n_uint16_frame*2)
    if mod != 0:
        incomplete_uint16_num = n_uint16_frame - mod//2
        adc_data              = adc_data[incomplete_uint16_num:]

    n_frame                 = adc_data.shape[0] // n_uint16_frame
    last_frame_start_index  = n_frame * n_uint16_frame
    adc_data                = adc_data[:last_frame_start_index]

    adc_data_complex = np.zeros(shape=(len(adc_data)//2), dtype=complex)
    adc_data_complex[0::2] = adc_data[0::4]+1j*adc_data[2::4]
    adc_data_complex[1::2] = adc_data[1::4]+1j*adc_data[3::4]
    

    frames = np.reshape(adc_data_complex, (n_frame, n_loop, n_virtual_ant, n_sample))
    if debug:
        print(f"adc_data_complex.shape: {adc_data_complex.shape}")
        print(f"frames.shape (n_frame, n_loop, n_virtual_ant, n_sample): {frames.shape}")

    return frames

def organize(adc_frames, layout, chirp_cfg, n_rx, debug=False):
    """
    adc_frames: (n_frame, n_loop, n_virtual_ant, n_sample)

    iwr1843boost:              Tx: (0, 1, 2)                                        ====>                           Tx: (0, 2, 1)
    
                          o       o      o       o                          0                                  o       o      o       o
                        T2-R1   T2-R2  T2-R3   T2-R4                                                         T2-R1   T2-R2  T2-R3   T2-R4
                         (4)     (5)    (6)     (7)                                                           (8)     (9)     (10)    (11)
          o       o       o       o      o       o       o       o          -1      ====>      o       o       o       o      o       o       o       o
        T1-R1   T1-R2   T1-R3   T1-R4  T3-R1   T3-R2   T3-R3   T3-R4                         T1-R1   T1-R2   T1-R3   T1-R4  T3-R1   T3-R2   T3-R3   T3-R4               
         (0)     (1)     (2)     (3)    (8)     (9)     (10)    (11)                          (0)     (1)     (2)     (3)    (4)     (5)    (6)     (7)
          0      -1      -2      -3     -4      -5      -6      -7                            0       -1      -2      -3     -4      -5      -6      -7     

    iwr6843aop:       Tx: (0, 1, 2)  
    
                o       o                            0             
              T1-R4   T1-R2
               (3)     (1)                                        
                o       o                           -1             
              T1-R3   T1-R1
               (2)     (0)                                        
                o       o       o       o           -2             
              T3-R4   T3-R2   T2-R4   T2-R2
               (11)    (9)     (7)     (5)                    
                o       o       o       o           -3             
              T3-R3   T3-R1   T2-R3   T2-R1                        
               (10)    (8)     (6)     (4)
                0       -1      -2      -3                                  
              
    """
    if layout == TI_RADAR_TYPE.iwr1843boost:
        if debug:
            print(f"chirp_cfg: {chirp_cfg} CHIRP_CFG: {CHIRP_CFG_DICT[layout]}")
        if not np.array_equal(chirp_cfg, CHIRP_CFG_DICT[layout]):
            order       = np.argsort(chirp_cfg)
            reordering  = np.array([n_rx*i+np.arange(n_rx) for i in order]).reshape(-1)
            adc_frames  = adc_frames[:, :, reordering]
            if debug:
                print(f"chirp_cfg: {chirp_cfg} \rreordering: {reordering}")

    elif layout == TI_RADAR_TYPE.iwr6843aop:
        if not np.array_equal(chirp_cfg, CHIRP_CFG_DICT[layout]):
            order       = np.argsort(chirp_cfg)
            reordering  = np.array([n_rx*i+np.arange(n_rx) for i in order]).reshape(-1)
            adc_frames  = adc_frames[:, :, reordering]
            if debug:
                print(f"chirp_cfg: {chirp_cfg} \rreordering: {reordering}")

    reordering  = REORDEING[layout]
    adc_frames  = adc_frames[:, :, reordering]
    if debug:
        print(f"CHIRP_CFG: {CHIRP_CFG_DICT[layout]} \rreordering: {reordering}")
    
    adc_frames = adc_frames * ANT_PHASE_ROTATE[layout][np.newaxis, np.newaxis, :, np.newaxis]
    return adc_frames

if __name__=="__main__":
    adc_frames_a = np.arange(12)[np.newaxis, np.newaxis, :, np.newaxis]
    adc_frames_b = organize(adc_frames_a, TI_RADAR_TYPE.iwr1843boost, np.array([2, 1, 0]), 4, debug=True)
    print(f"adc_frames_a.shape: {adc_frames_a.shape} adc_frames_b.shape: {adc_frames_b.shape}")
    print(f"adc_frames_a: {adc_frames_a} adc_frames_b: {adc_frames_b}")
