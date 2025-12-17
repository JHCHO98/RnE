document.addEventListener('DOMContentLoaded', function () {
    const btn = document.getElementById("collectBtn");

    if (!btn) {
        console.error("오류: 버튼을 찾을 수 없습니다.");
        return;
    }

    btn.addEventListener("click", async () => {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        if (!tab.url.includes("youtube.com")) {
            alert("유튜브 페이지에서 실행해주세요!");
            return;
        }

        // 1. 스크립트 실행 (파견)
        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            function: scrapeVideoIds, // 수집 함수 실행
        }, (results) => {
            // 2. 결과 수신 (본부 복귀)
            if (chrome.runtime.lastError) {
                console.error("실행 에러:", chrome.runtime.lastError);
                return;
            }

            // 결과값 확인 및 전송
            if (results && results[0] && results[0].result) {
                const collectedIds = results[0].result;
                console.log(`Algo.Analyzer: 본부 수신 완료 (${collectedIds.length}개). 리액트로 전송합니다.`);

                // ★ 여기서 전송 함수를 호출해야 합니다! (본부에서 실행)
                sendDataToReactApp(collectedIds);
            } else {
                console.warn("데이터를 가져오지 못했습니다.");
            }
        });
    });
});

// --- [A] 유튜브 페이지에서 실행될 함수 (수집 및 반환만 함) ---
function scrapeVideoIds() {
    console.log("Algo.Analyzer: 1. 수집 시작...");

    const allLinks = document.querySelectorAll('a');
    const videoIds = new Set();

    allLinks.forEach((link) => {
        const href = link.href;
        if (!href) return;

        try {
            if (href.includes('/watch?v=')) {
                const id = new URL(href).searchParams.get('v');
                if (id && id.length > 5) videoIds.add(id);
            } else if (href.includes('/shorts/')) {
                const id = href.split('/shorts/')[1]?.split('?')[0];
                if (id && id.length > 5) videoIds.add(id);
            }
        } catch (e) { }
    });

    const uniqueIds = Array.from(videoIds);
    const finalIds = uniqueIds.slice(0, 1000);

    console.log(`Algo.Analyzer: 2. 수집 완료 (${finalIds.length}개). 본부로 반환합니다.`);

    // ★ 중요: 직접 전송하지 않고, 결과값만 리턴합니다!
    return finalIds;
}

// --- [B] 리액트 앱으로 데이터 주입하는 함수 (popup.js 내부에서 실행됨) ---
// --- [B] 리액트 앱으로 데이터 강제 주입 및 새로고침 ---
function sendDataToReactApp(videoIds) {
    const targetUrl = "http://localhost:3000";

    // 1. 새 탭 열기
    chrome.tabs.create({ url: targetUrl }, (newTab) => {

        // 2. 탭이 다 로딩될 때까지 기다림
        chrome.tabs.onUpdated.addListener(function listener(tabId, info) {
            if (tabId === newTab.id && info.status === 'complete') {
                chrome.tabs.onUpdated.removeListener(listener);

                console.log(`Algo.Analyzer: 3. 데이터 ${videoIds.length}개 주입 시작...`);

                // 3. 스크립트 주입 (기존 데이터 삭제 -> 새 데이터 저장 -> 새로고침)
                chrome.scripting.executeScript({
                    target: { tabId: newTab.id },
                    func: (data) => {
                        // [중요] 옛날 찌꺼기 데이터 삭제
                        localStorage.removeItem("youtube_data_ids");

                        // 새 데이터 저장
                        localStorage.setItem("youtube_data_ids", JSON.stringify(data));

                        // [중요] 리액트가 새 데이터를 물도록 강제 새로고침
                        // alert("새로운 데이터 900개가 도착했습니다! 확인을 누르면 분석을 시작합니다."); // 확인용 알림
                        window.location.reload();
                    },
                    args: [videoIds] // 수집한 900개 데이터 전달
                });
            }
        });
    });
}