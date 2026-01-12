import os
import geopandas as gpd

from attacks1_vertex_delete_poly import attacks1_vertex_delete
from attacks2_vertex_add import attacks2_vertex_add
from attacks3_vertex_noise import attacks3_vertex_noise
from attacks4_object_delete import attacks4_object_delete
from attacks6_1_vertex_reorganization import attacks6_1_vertex_reorganization, attacks6_2_object_reorganization
from attacks7_geometric import attacks7_geometric


def attacks8_compound(embedshp, outshpfile):
    """
    执行组合攻击，包含多种攻击方式的组合
    
    参数:
        embedshp: 嵌入水印的shapefile路径
        outshpfile: 输出文件名
    
    返回:
        最终攻击后的文件路径
    """
    # 创建保存复合攻击结果的目录
    output_dir = os.path.join('attacked', 'compound')
    os.makedirs(output_dir, exist_ok=True)

    # 开始执行复合攻击
    print('🚀 开始执行组合攻击...')
    print('=' * 50)

    try:
        # 1. 顶点删除攻击（删除30%的顶点）
        print('1. 执行顶点删除攻击 (删除率=0.3)')
        c_vertex_delete = 0.3
        temp_name1 = f'step1_vertex_delete_{outshpfile}'
        tempshp1 = attacks1_vertex_delete(embedshp, temp_name1, c_vertex_delete)
        print(f'   完成，输出: {tempshp1}')

        # 2. 对象删除攻击（删除30%的对象）
        print('2. 执行对象删除攻击 (删除率=0.3)')
        c_object_delete = 0.3
        temp_name2 = f'step2_object_delete_{outshpfile}'
        tempshp2 = attacks4_object_delete(tempshp1, temp_name2, c_object_delete)
        print(f'   完成，输出: {tempshp2}')

        # 3. 顶点增加攻击（增加10%的顶点）
        print('3. 执行顶点增加攻击 (增加率=0.1, 强度=1)')
        c_vertex_add = 0.1
        strength_add = 1.0
        tolerance = 0.01
        temp_name3 = f'step3_vertex_add_{outshpfile}'
        tempshp3 = attacks2_vertex_add(
            watermarkedshp=tempshp2,
            outshpfile=temp_name3,
            addRatio=c_vertex_add,
            strength=strength_add,
            tolerance=tolerance
        )
        print(f'   完成，输出: {tempshp3}')

        # 4. 顶点噪声攻击（噪声比例20%，强度0.6）
        print('4. 执行顶点噪声攻击 (比例=0.2, 强度=0.6)')
        c_vertex_noise = 0.1
        noise_strength = 0.6
        temp_name4 = f'step4_vertex_noise_{outshpfile}'
        tempshp4 = attacks3_vertex_noise(tempshp3, temp_name4, c_vertex_noise, noise_strength)
        print(f'   完成，输出: {tempshp4}')

        # 5. 几何攻击：平移（X和Y方向各平移10单位）
        print('5. 执行几何攻击：平移 (x=10, y=10)')
        x_shift = 10
        y_shift = 10
        temp_name5 = f'step5_translation_{outshpfile}'
        tempshp5 = attacks7_geometric(tempshp4, temp_name5, 0, 1, x_shift, y_shift)
        print(f'   完成，输出: {tempshp5}')

        # 6. 几何攻击：缩放（缩放因子0.5）
        print('6. 执行几何攻击：缩放 (因子=0.5)')
        scale_factor = 0.5
        temp_name6 = f'step6_scaling_{outshpfile}'
        tempshp6 = attacks7_geometric(tempshp5, temp_name6, 0, scale_factor, 0, 0)
        print(f'   完成，输出: {tempshp6}')

        # 7. 几何攻击：旋转（旋转0度）
        print('7. 执行几何攻击：旋转 (角度=0°)')
        angle = 0
        temp_name7 = f'step7_rotation_{outshpfile}'
        tempshp7 = attacks7_geometric(tempshp6, temp_name7, angle, 1, 0, 0)
        print(f'   完成，输出: {tempshp7}')

        # 8. 顶点重组攻击
        print('8. 执行顶点重组攻击')
        temp_name8 = f'step8_vertex_reorg_{outshpfile}'
        tempshp8 = attacks6_1_vertex_reorganization(tempshp7, temp_name8)
        print(f'   完成，输出: {tempshp8}')

        # 9. 对象重组攻击
        print('9. 执行对象重组攻击')
        final_name = f'compound_{outshpfile}'
        finalshp = attacks6_2_object_reorganization(tempshp8, final_name)
        print(f'   完成，输出: {finalshp}')

        print('=' * 50)
        print(f'✅ 组合攻击完成！最终结果: {finalshp}')
        
        return finalshp

    except Exception as e:
        print(f'❌ 组合攻击执行失败: {str(e)}')
        raise e
