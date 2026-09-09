from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from supabase import create_client, Client
from ortools.sat.python import cp_model

app = FastAPI(title="Emoteca Solver API", version="1.0")

# Connessione a Supabase tramite variabili d'ambiente
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=500, detail="Credenziali Supabase non configurate nel server Python.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

class OperatoreInput(BaseModel):
    id: str
    nome: str
    ruolo: Optional[str] = "Operatore"
    stato_disponibilita: str
    in_turno_pomeriggio: bool

class TurnoRequest(BaseModel):
    data_turno: str
    anno: int
    operatori: List[OperatoreInput]

@app.get("/")
def read_root():
    return {"status": "online", "service": "Emoteca Solver Python API"}

@app.post("/calcola-turno")
def calcola_turno(payload: TurnoRequest):
    supabase = get_supabase_client()
    
    # 1. Interroghiamo direttamente la tabella 'turni' su Supabase per la data specifica
    try:
        response_turni = supabase.table("turni").select("utente_id, fascia_oraria").eq("data", payload.data_turno).execute()
        turni_ospedale = response_turni.data if response_turni and response_turni.data else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore lettura turni ospedale da Supabase: {str(e)}")

    # Filtriamo gli utenti che hanno la fascia oraria del pomeriggio (gestendo eventuali variazioni di maiuscole/minuscole)
    utenti_in_turno_pomeriggio = set()
    for t in turni_ospedale:
        fascia = str(t.get("fascia_oraria", "")).strip().lower()
        if fascia in ["pomeriggio", "p", "pom"]:
            if t.get("utente_id"):
                utenti_in_turno_pomeriggio.add(str(t.get("utente_id")))

    # 2. Filtriamo gli operatori idonei tra quelli inviati (disponibili E in turno pomeriggio su Supabase)
    operatori_idonei = [
        op for op in payload.operatori 
        if op.stato_disponibilita == 'disponibile' and str(op.id) in utenti_in_turno_pomeriggio
    ]

    # Fallback di sicurezza: se per quella data non ci sono turni inseriti nella tabella 'turni', consideriamo tutti i disponibili
    if not operatori_idonei and not turni_ospedale:
        operatori_idonei = [op for op in payload.operatori if op.stato_disponibilita == 'disponibile']

    if not operatori_idonei:
        raise HTTPException(status_code=400, detail=f"Nessun operatore idoneo (disponibile e in turno 'Pomeriggio') per la data {payload.data_turno}.")

    # 3. Recuperiamo lo storico annuale dei turni_trasporti da Supabase per calcolare l'equità
    inizio_anno = f"{payload.anno}-01-01"
    try:
        response_storico = supabase.table("turni_trasporti").select("utente_id").gte("data_turno", inizio_anno).lte("data_turno", payload.data_turno).execute()
        storico_data = response_storico.data if response_storico and response_storico.data else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore lettura storico da Supabase: {str(e)}")

    # Contiamo quanti turni ha fatto ciascuno dall'inizio dell'anno
    conteggio_turni = {op.id: 0 for op in operatori_idonei}
    for item in storico_data:
        uid = str(item.get("utente_id"))
        if uid in conteggio_turni:
            conteggio_turni[uid] += 1

    # 4. Utilizziamo Google OR-Tools per trovare la soluzione ottimale (minimizzare il carico di lavoro)
    model = cp_model.CpModel()
    
    x = {}
    for op in operatori_idonei:
        x[op.id] = model.NewBoolVar(f"x_{op.id}")

    model.Add(sum(x[op.id] for op in operatori_idonei) == 1)

    objective_terms = []
    for op in operatori_idonei:
        carico_attuale = conteggio_turni[op.id]
        objective_terms.append(x[op.id] * carico_attuale)

    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for op in operatori_idonei:
            if solver.Value(x[op.id]) == 1:
                return {
                    "id_utente_selezionato": op.id,
                    "nome_selezionato": op.nome,
                    "turni_pregressi_anno": conteggio_turni[op.id],
                    "motivazione": f"Selezionato tramite OR-Tools in base all'equità annuale (turni precedenti: {conteggio_turni[op.id]})"
                }

    raise HTTPException(status_code=500, detail="Impossibile trovare una soluzione di assegnazione valida tramite il solver.")
