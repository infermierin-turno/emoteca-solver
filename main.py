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
    
    # 1. Filtriamo solo gli operatori idonei per il giorno (disponibili e in turno di pomeriggio)
    operatori_idonei = [
        op for op in payload.payload_operatori if op.stato_disponibilita == 'disponibile' and op.in_turno_pomeriggio
    ] if hasattr(payload, 'payload_operatori') else [
        op for op in payload.operatori if op.stato_disponibilita == 'disponibile' and op.in_turno_pomeriggio
    ]

    if not operatori_idonei:
        raise HTTPException(status_code=400, detail="Nessun operatore idoneo (disponibile e in turno pomeridiano) per questa data.")

    # 2. Recuperiamo lo storico annuale dei turni_trasporti da Supabase per calcolare l'equità
    inizio_anno = f"{payload.anno}-01-01"
    try:
        response = supabase.table("turni_trasporti").select("utente_id").gte("data_turno", inizio_anno).lte("data_turno", payload.data_turno).execute()
        storico_data = response.data if response and response.data else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore lettura storico da Supabase: {str(e)}")

    # Contiamo quanti turni ha fatto ciascuno dall'inizio dell'anno
    conteggio_turni = {op.id: 0 for op in operatori_idonei}
    for item in storico_data:
        uid = item.get("utente_id")
        if uid in conteggio_turni:
            conteggio_turni[uid] += 1

    # 3. Utilizziamo Google OR-Tools per trovare la soluzione ottimale (minimizzare il carico di lavoro)
    model = cp_model.CpModel()
    
    # Variabile binaria per ciascun operatore idoneo (1 = viene scelto, 0 = no)
    x = {}
    for op in operatori_idonei:
        x[op.id] = model.NewBoolVar(f"x_{op.id}")

    # Vincolo: dobbiamo assegnare esattamente 1 operatore per questo turno
    model.Add(sum(x[op.id] for op in operatori_idonei) == 1)

    # Funzione obiettivo: minimizzare la sperequazione dando priorità a chi ha fatto meno turni
    # Assegniamo un peso inversamente proporzionale o penalizziamo direttamente il conteggio storico
    objective_terms = []
    for op in operatori_idonei:
        carico_attuale = conteggio_turni[op.id]
        # Il solver cercherà di minimizzare la somma pesata (chi ha più turni ha un costo maggiore)
        objective_terms.append(x[op.id] * carico_attuale)

    model.Minimize(sum(objective_terms))

    # Eseguiamo il solver
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
