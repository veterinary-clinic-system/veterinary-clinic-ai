import sys
from src.db import get_db_connection

symptoms_data = [
    ('SY001', 'Chưa tiêm vaccine', 'Thú cưng chưa hoàn thành phác đồ vaccine cơ bản', True),
    ('SY002', 'Ăn bậy / Nuốt dị vật', 'Ăn phải thức ăn lạ, rác hoặc đồ chơi dị vật', True),
    ('SY003', 'Tiếp xúc động vật bệnh', 'Có tiền sử tiếp xúc với vật nuôi bị bệnh truyền nhiễm', False),
    ('SY004', 'Co giật / Run rẩy', 'Các cơn co giật toàn thân hoặc co giật cơ cục bộ', True),
    ('SY005', 'Sốt nhẹ', 'Thân nhiệt từ 39.2C đến 39.7C', False),
    ('SY006', 'Hạ thân nhiệt', 'Thân nhiệt dưới 37.5C, chân tay lạnh', False),
    ('SY007', 'Sốt vừa', 'Thân nhiệt từ 39.8C đến 40.3C', False),
    ('SY008', 'Sốt cao kéo dài', 'Thân nhiệt trên 40.5C liên tục nhiều ngày', True),
    ('SY009', 'Hôn mê / Bất tỉnh', 'Không phản ứng với kích thích âm thanh hoặc xúc giác', True),
    ('SY010', 'Yếu sức / Suy nhược nặng', 'Không thể tự đứng dậy, kiệt sức rõ rệt', True),
    ('SY011', 'Bỏ ăn / Chán ăn', 'Từ chối thức ăn liên tục trên 24-48 giờ', True),
    ('SY012', 'Mệt mỏi / Lừ đừ', 'Giảm vận động, nằm nhiều một chỗ', True),
    ('SY013', 'Dị ứng da / Nổi mẩn đỏ', 'Da nổi sẩn, đỏ ửng hoặc phát ban mề đay', True),
    ('SY014', 'Viêm tai / Tai có mủ', 'Tai đỏ rực, chảy dịch mủ có mùi hôi', True),
    ('SY015', 'Rụng lông cục bộ', 'Lông rụng thành mảng tròn hoặc loang lổ', False),
    ('SY016', 'Vảy gàu / Da bong tróc', 'Lớp vảy sừng dày, bong tróc trên bề mặt da', False),
    ('SY017', 'Ngứa da dữ dội', 'Gãi, cọ xát cơ thể liên tục gây trầy xước', True),
    ('SY018', 'Chảy nước mắt / Ghèn mắt', 'Mắt đỏ, tiết nhiều ghèn mủ vàng xanh', False),
    ('SY019', 'Chảy nước mũi / Dịch mũi đục', 'Dịch nhầy hoặc mủ đặc chảy từ mũi', False),
    ('SY020', 'Hắt hơi liên tục', 'Hắt hơi theo tràng, dụi mũi', False),
    ('SY021', 'Khó thở / Thở khò khè', 'Thở há miệng, co kéo lồng ngực và cơ bụng', True),
    ('SY022', 'Ho khạc / Ho khan', 'Ho từng cơn, có thể khạc bọt trắng', False),
    ('SY023', 'Thở nhanh / Thở dốc', 'Nhịp thở tăng cao bất thường khi nghỉ ngơi', True),
    ('SY024', 'Uống nhiều nước / Khát nước quá mức', 'Lượng nước uống vượt quá 100ml/kg/ngày', False),
    ('SY025', 'Nôn mửa', 'Nôn ra thức ăn, dịch mật vàng hoặc dịch bọt', True),
    ('SY026', 'Tiêu chảy / Phân lỏng', 'Phân lỏng nhiều nước, đi ngoài nhiều lần', True),
    ('SY027', 'Tiêu chảy có máu / Phân đen', 'Phân tanh khắm có máu tươi hoặc phân đen như hắc ín', True),
    ('SY028', 'Bụng phình to / Chướng hơi', 'Vòng bụng căng cứng, gõ có tiếng vang', True),
    ('SY029', 'Tiểu nhiều lần / Tiểu rắt', 'Đi tiểu liên tục, mỗi lần ra rất ít', False),
    ('SY030', 'Tiểu ra máu', 'Nước tiểu có màu đỏ hồng hoặc nâu sẫm', True),
    ('SY031', 'Bí tiểu / Không đi tiểu được', 'Rặn tiểu đau đớn, bàng quang căng to', True),
    ('SY032', 'Đau đớn khi chạm vào / Kêu la', 'Phản ứng dữ dội, cắn hoặc rên rỉ khi chạm vào vùng đau', True),
]

diseases_data = [
    ('DI001', 'Viêm dạ dày ruột', 'Viêm cấp tính niêm mạc dạ dày và ruột non'),
    ('DI002', 'Suy thận cấp', 'Tổn thương chức năng thận cấp tính gây ứ đọng ure huyết'),
    ('DI003', 'Bệnh Care (Canine Distemper)', 'Bệnh truyền nhiễm do virus Morbillivirus ở chó'),
    ('DI004', 'Bệnh Parvovirus', 'Viêm ruột xuất huyết cấp tính do Canine Parvovirus'),
    ('DI005', 'Viêm gan truyền nhiễm', 'Bệnh viêm gan do Canine Adenovirus type 1'),
    ('DI006', 'Ho cũi chó (Kennel Cough)', 'Hội chứng viêm phế quản truyền nhiễm ở chó'),
    ('DI007', 'Bệnh Lepto (Leptospirosis)', 'Bệnh truyền nhiễm lây sang người do xoắn khuẩn Leptospira'),
    ('DI008', 'Giảm bạch cầu mèo (FPV)', 'Viêm ruột truyền nhiễm nặng do Feline Panleukopenia Virus'),
    ('DI009', 'Viêm mũi khí quản truyền nhiễm (FHV-1)', 'Bệnh hô hấp trên ở mèo do Feline Herpesvirus-1'),
    ('DI010', 'Sốt sữa / Hạ canxi huyết sau sinh', 'Sốt cao và co giật ở chó mèo mẹ sau khi sinh con'),
    ('DI011', 'Viêm tụy cấp', 'Viêm hoại tử tuyến tụy do kích hoạt sớm enzym tiêu hóa'),
    ('DI012', 'Viêm phổi / Nhiễm trùng hô hấp dưới', 'Nhiễm trùng đường hô hấp dưới gây sốt và khó thở'),
    ('DI013', 'Nhiễm trùng đường tiết niệu (UTI)', 'Viêm bàng quang và đường dẫn tiểu do vi khuẩn'),
    ('DI014', 'Sỏi bàng quang / Tắc niệu đạo', 'Sỏi kết tinh gây tắc nghẽn đường tiểu'),
    ('DI015', 'Dị ứng da / Viêm da cơ địa', 'Phản ứng quá mẫn với thức ăn hoặc môi trường'),
    ('DI016', 'Nhiễm nấm da (Ringworm)', 'Nhiễm vi nấm Microsporum canis gây rụng lông tròn'),
    ('DI017', 'Bệnh ghẻ da (Demodex / Sarcoptes)', 'Bệnh do ký sinh trùng ve bao lông hoặc ghẻ ngứa'),
    ('DI018', 'Viêm tai ngoài (Otitis Externa)', 'Nhiễm trùng ống tai ngoài do nấm men hoặc vi khuẩn'),
    ('DI019', 'Bệnh dại (Rabies)', 'Bệnh truyền nhiễm virus cấp tính tấn công hệ thần kinh'),
    ('DI020', 'Suy tim sung huyết', 'Suy giảm khả năng bơm máu của tim gây phù phổi'),
    ('DI021', 'Đái tháo đường', 'Rối loạn chuyển hóa do thiếu hụt insulin'),
    ('DI022', 'Hội chứng Cushing', 'Cường vỏ thượng thận do tăng tiết cortisol'),
    ('DI023', 'Bệnh Calicivirus (FCV)', 'Bệnh hô hấp trên kèm loét miệng ở mèo'),
    ('DI024', 'Bệnh Chlamydia mèo', 'Viêm kết mạc mắt và đường hô hấp trên do Chlamydia'),
    ('DI025', 'Nhiễm ký sinh trùng máu (Babesia)', 'Ký sinh trùng phá hủy hồng cầu gây thiếu máu và vàng da'),
    ('DI026', 'Bệnh giun tim (Dirofilaria)', 'Ký sinh trùng cư trú trong động mạch phổi và tâm thất phải'),
    ('DI027', 'Tắc ruột do dị vật', 'Dị vật bít tắc hoàn toàn hoặc một phần lòng ruột'),
    ('DI028', 'Xoắn dạ dày chướng hơi (GDV)', 'Cấp cứu ngoại khoa khi dạ dày xoắn nghẹt'),
    ('DI029', 'Viêm tử cung tích mủ (Pyometra)', 'Nhiễm trùng tích mủ trong lòng tử cung ở thú cái'),
    ('DI030', 'Chấn thương / Gãy xương', 'Tổn thương cơ xương khớp do tai nạn va chạm'),
    ('DI031', 'Ngộ độc cấp tính', 'Nhiễm độc qua đường tiêu hóa hoặc hô hấp'),
    ('DI032', 'Khám sức khỏe tổng quát định kỳ', 'Kiểm tra lâm sàng định kỳ đánh giá tình trạng sức khỏe chung'),
]

conn = get_db_connection()
cur = conn.cursor()

# 1. Create symptoms table
cur.execute("""
    CREATE TABLE IF NOT EXISTS symptoms (
        symptom_id VARCHAR(50) PRIMARY KEY,
        symptom_name VARCHAR(255) NOT NULL,
        describe TEXT,
        common_symptom BOOLEAN DEFAULT FALSE
    );
""")

# 2. Add columns to diseases if missing
cur.execute("""
    ALTER TABLE diseases ADD COLUMN IF NOT EXISTS disease_id VARCHAR(50);
    ALTER TABLE diseases ADD COLUMN IF NOT EXISTS describe TEXT;
""")

# 3. Seed symptoms
for sid, sname, sdesc, scomm in symptoms_data:
    cur.execute("""
        INSERT INTO symptoms (symptom_id, symptom_name, describe, common_symptom)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (symptom_id) DO UPDATE
        SET symptom_name = EXCLUDED.symptom_name,
            describe = EXCLUDED.describe,
            common_symptom = EXCLUDED.common_symptom;
    """, (sid, sname, sdesc, scomm))

# 4. Map or insert diseases
for did, dname, ddesc in diseases_data:
    cur.execute("""
        SELECT id FROM diseases WHERE disease_id = %s OR disease_name ILIKE %s LIMIT 1;
    """, (did, f"%{dname[:8]}%"))
    row = cur.fetchone()
    if row:
        cur.execute("""
            UPDATE diseases SET disease_id = %s, describe = %s WHERE id = %s;
        """, (did, ddesc, row[0]))
    else:
        cur.execute("""
            INSERT INTO diseases (id, disease_id, disease_name, describe, "commonSymptoms", other_symptoms, created_at, updated_at)
            VALUES (gen_random_uuid(), %s, %s, %s, '{}'::diseases_commonsymptoms_enum[], NULL, NOW(), NOW());
        """, (did, dname, ddesc))

conn.commit()
print(f"SUCCESS: Seeded {len(symptoms_data)} symptoms and {len(diseases_data)} diseases.")
conn.close()
