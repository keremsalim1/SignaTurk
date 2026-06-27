# ═══════════════════════════════════════════════════════════
#  ROUTES — 3D Animasyon (senin endpoint'lerin)
# ═══════════════════════════════════════════════════════════
@app.get("/signs")
async def list_signs():
    """Mevcut tüm işaret kelimelerini listele."""
    return {"words": sorted(LANDMARK_INDEX.keys()), "count": len(LANDMARK_INDEX)}

@app.get("/landmark/{word}")
async def get_landmark(word: str):
    """Kelimeye ait landmark verisini döndür (smooth edilmiş)."""
    key = word.lower().strip()
    data = LANDMARK_INDEX.get(key)
    if data is None:
        return Response(status_code=404, content=f"'{word}' bulunamadı")
    if SMOOTHER_AVAILABLE and not data.get("smoothed"):
        data = smooth_landmark_data(data)
    return Response(
        content=json.dumps(data, ensure_ascii=False),
        media_type="application/json"
    )

# ═══════════════════════════════════════════════════════════
#  ROUTES — Auth
# ═══════════════════════════════════════════════════════════
@app.post("/api/auth/register")
def register(data: RegisterData, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == data.email.lower()).first():
        return JSONResponse(status_code=400, content={"error": "Email zaten kayıtlı"})
    user = models.User(
        full_name=data.full_name, email=data.email.lower(),
        password_hash=hash_pw(data.password),
        role="User", status="Active", sessions=0,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    add_log_db(db, "Info", f"Yeni kullanıcı: {data.email}")
    return {"message": "Kayıt başarılı",
            "user": {"id": user.id, "full_name": user.full_name,
                     "email": user.email, "role": user.role}}

@app.post("/api/auth/login")
def login(data: LoginData, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == data.email.lower()).first()
    if not user or not verify_pw(data.password, user.password_hash):
        return JSONResponse(status_code=401, content={"error": "Hatalı email veya şifre"})
    user.sessions += 1
    db.commit()
    add_log_db(db, "Success", f"Giriş: {user.email}")
    return {"message": "Giriş başarılı",
            "user": {"id": user.id, "full_name": user.full_name,
                     "email": user.email, "role": user.role}}


# ═══════════════════════════════════════════════════════════
#  ROUTES — Canlı tahmin WebSocket (tsl-nexus)
# ═══════════════════════════════════════════════════════════
@app.websocket("/api/predict/live-legacy")
async def live_predict(websocket: WebSocket):