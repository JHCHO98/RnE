// 익스텐션이 설치될 때 메뉴 생성
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "analyze-selection",
        title: "⚡ Hyperspeed 분석 실행", // 우클릭 메뉴에 뜰 이름
        contexts: ["selection"] // 텍스트를 드래그했을 때만 메뉴가 나옴
    });
});

// 메뉴 클릭 이벤트 리스너
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "analyze-selection" && info.selectionText) {

        // 1. 선택된 텍스트 인코딩 (URL에 넣기 위해)
        const text = encodeURIComponent(info.selectionText);

        // 2. 리액트 앱 주소 (개발 중엔 localhost, 배포 후엔 실제 도메인으로 변경)
        // 쿼리 파라미터(?q=...)로 데이터를 전달합니다.
        const targetUrl = `http://localhost:3000?q=${text}`;

        // 3. 새 탭 열기
        chrome.tabs.create({ url: targetUrl });
    }
});