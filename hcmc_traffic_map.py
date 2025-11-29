import streamlit as st
import osmnx as ox
import folium
from streamlit_folium import st_folium
import pandas as pd
import warnings

# Ẩn cảnh báo
warnings.filterwarnings('ignore')

# Cấu hình trang
st.set_page_config(
    page_title="Bản Đồ Giao Thông TP.HCM",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ẩn các phần tử không cần thiết
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


class HCMTrafficMap:
    def __init__(self):
        ox.settings.use_cache = True
        ox.settings.log_console = False
        ox.settings.timeout = 300

    def load_all_roads(self):
        """Tải tất cả các đường trong TP.HCM"""
        try:
            # Tạo placeholder cho tiến trình
            progress_placeholder = st.empty()
            progress_placeholder.info("🔄 Đang tải dữ liệu đường từ OpenStreetMap... Vui lòng chờ (có thể mất vài phút)")

            # Tải dữ liệu đường bộ cho toàn TP.HCM
            G = ox.graph_from_place(
                "Ho Chi Minh City, Vietnam",
                network_type='drive',
                simplify=True
            )

            # Chuyển đổi thành GeoDataFrame
            nodes, edges = ox.graph_to_gdfs(G)

            progress_placeholder.empty()
            st.success(f"✅ Đã tải thành công {len(edges)} tuyến đường")

            return edges

        except Exception as e:
            st.error(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
            return None

    def create_traffic_map(self, edges):
        """Tạo bản đồ giao thông tương tác"""
        # Tạo bản đồ với tâm là TP.HCM
        m = folium.Map(
            location=[10.8231, 106.6297],
            zoom_start=12,
            tiles='OpenStreetMap'
        )

        # Màu sắc cho các loại đường
        highway_colors = {
            'motorway': '#FF0000',  # Đỏ - Đường cao tốc
            'trunk': '#FF4500',  # Cam đỏ - Quốc lộ
            'primary': '#FFA500',  # Cam - Tỉnh lộ
            'secondary': '#FFFF00',  # Vàng - Đường liên huyện
            'tertiary': '#00FF00',  # Xanh lá - Đường đô thị
            'residential': '#0000FF',  # Xanh dương - Đường nội bộ
            'unclassified': '#808080',  # Xám - Đường chưa phân loại
            'service': '#A9A9A9',  # Xám đậm - Đường dịch vụ
        }

        # Độ dày đường
        highway_weights = {
            'motorway': 6,
            'trunk': 5,
            'primary': 4,
            'secondary': 3,
            'tertiary': 3,
            'residential': 2,
            'unclassified': 2,
            'service': 1,
        }

        # Thêm các đường vào bản đồ
        for idx, row in edges.iterrows():
            try:
                # Lấy loại đường
                highway_type = row.get('highway', 'unclassified')
                if isinstance(highway_type, list):
                    highway_type = highway_type[0] if highway_type else 'unclassified'

                # Chọn màu và độ dày
                color = highway_colors.get(highway_type, '#808080')
                weight = highway_weights.get(highway_type, 1)

                # Tên đường
                road_name = row.get('name', 'Đường không tên')
                if pd.isna(road_name):
                    road_name = 'Đường không tên'

                # Thông tin popup
                popup_text = f"""
                <b>{road_name}</b><br>
                <i>Loại đường: {highway_type}</i><br>
                <small>Chiều dài: {row.get('length', 0):.0f}m</small>
                """

                # Vẽ đường trên bản đồ
                if hasattr(row.geometry, 'coords'):
                    folium.PolyLine(
                        locations=[(lat, lon) for lon, lat in row.geometry.coords],
                        popup=folium.Popup(popup_text, max_width=300),
                        color=color,
                        weight=weight,
                        opacity=0.8,
                        tooltip=f"{road_name} ({highway_type})"
                    ).add_to(m)

            except Exception:
                continue

        # Thêm chú thích
        self._add_legend(m)

        return m

    def _add_legend(self, map_obj):
        """Thêm chú thích cho bản đồ"""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; 
                    background-color: white; 
                    border: 2px solid grey; 
                    z-index: 9999; 
                    font-size: 14px; 
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.2);
                    width: 300px;">
            <h4 style="margin: 0 0 10px 0; text-align: center;">🏙️ Bản Đồ Giao Thông TP.HCM</h4>
            <p style="margin: 5px 0;"><span style="color: #FF0000; font-weight: bold;">━━━━━</span> Đường cao tốc</p>
            <p style="margin: 5px 0;"><span style="color: #FF4500; font-weight: bold;">━━━━━</span> Quốc lộ</p>
            <p style="margin: 5px 0;"><span style="color: #FFA500; font-weight: bold;">━━━━━</span> Tỉnh lộ</p>
            <p style="margin: 5px 0;"><span style="color: #FFFF00; font-weight: bold;">━━━━━</span> Đường liên huyện</p>
            <p style="margin: 5px 0;"><span style="color: #00FF00; font-weight: bold;">━━━━━</span> Đường đô thị</p>
            <p style="margin: 5px 0;"><span style="color: #0000FF; font-weight: bold;">━━━━━</span> Đường nội bộ</p>
            <p style="margin: 10px 0 0 0; font-size: 12px; color: #666; text-align: center;">
                Dữ liệu từ OpenStreetMap<br>
                Click vào đường để xem thông tin
            </p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))


def main():
    # Tiêu đề
    st.markdown("""
    <h1 style='text-align: center; color: #1f77b4; margin-bottom: 20px;'>
    🗺️ BẢN ĐỒ GIAO THÔNG THÀNH PHỐ HỒ CHÍ MINH
    </h1>
    """, unsafe_allow_html=True)

    # Khởi tạo ứng dụng
    app = HCMTrafficMap()

    # Tải dữ liệu
    if 'edges' not in st.session_state:
        edges = app.load_all_roads()
        if edges is not None:
            st.session_state.edges = edges
        else:
            st.stop()

    # Tạo và hiển thị bản đồ
    with st.spinner("🔄 Đang tạo bản đồ..."):
        traffic_map = app.create_traffic_map(st.session_state.edges)

        if traffic_map:
            # Hiển thị bản đồ với kích thước lớn
            st_folium(
                traffic_map,
                width=1400,
                height=700,
                returned_objects=[]
            )

            # Hiển thị thông tin
            st.markdown("""
            <div style='text-align: center; color: #666; margin-top: 20px;'>
            <p><strong>Hướng dẫn:</strong> Click vào các đường để xem thông tin chi tiết</p>
            <p><strong>Nguồn dữ liệu:</strong> OpenStreetMap © Contributors</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()