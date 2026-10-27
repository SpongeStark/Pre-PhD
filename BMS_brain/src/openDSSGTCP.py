# -*- coding: utf-8 -*-
"""
OpenDSS-G Direct Server TCP (DSTCP) Client Library for Python 3
===============================================================
Implements the EPRI OpenDSS-G DSTCP ASCII/Binary TCP protocol on port 6345
(per "Protocol for Data Exchange with OpenDSS-G - DSTCP", Davis Montenegro).

Compatible with the classic `openDSSGTCP` interface used in OpenDSS-G examples
while supporting Python 3 strings, full framing (`LLLLLLLL` hex length header),
and locale-safe float parsing (handling both `.` and `,` decimal separators).
"""

import socket


COMMANDS = {
    'seconds': '01;',
    'Close': '&Q;',
    'days': '02;',
    'Vpu': '03;',
    'V': '04;',
    'CapC': '05;',
    'CapV': '06;',
    'CapL': '07;',
    'CapP': '08;',
    'XfrtC': '09;',
    'XfrtV': '10;',
    'XfrtL': '11;',
    'XfrtP': '12;',
    'LoadC': '13;',
    'LoadP': '14;',
    'LineC': '15;',
    'LineP': '16;',
    'LineL': '17;',
    'SwitchC': '18;',
    'Switch': '19;',
    'FixPower': '1A;',
    'FixElement': '1B;',
    'EnergyM': '1C;',
    'Inc_Mat': '1D;',
    'EVStorage_List': '1E;',
    'EVStorage_Status': '1F;',
    'SW_Currents': '20;',
    'SW_Losses': '21;',
    'SW_Powers': '22;',
    'Step_in': '23;',
    'End_Sim': '24;',
    'Command': '25;',
    'Monitors': '26;',
    'Mon_data': '27;',
    'SetLineList': '28;',
    'SetLineColors': '29;',
    'SetColorMode': '30;',
    'RefreshGUI': '31;',
    'SelectedElems': '32;',
    'LineNames': '33;',
    'LineRatings': '34;',
    'ShowOvLoads': '36;',
    'ShowVolt': '37;',
    'ShowPower': '38;',
    'ClearCtrlQueue': '3F;',
    'GetCtrlQueue': '40;',
    'ShowIsolated': '41;',
    'BusNames': '42;',
    'NodeNames': '43;',
    'LoadNames': '44;',
    'XfrtNames': '45;',
    'CapNames': '46;',
    'StorageNames': '47;',
    'PVNames': '4A;',
    'GenNames': '4B;',
    'EngineStatus': '52;',
    'EngineProgress': '53;',
}

ONOFF = {
    'on': '1',
    'off': '0',
    'close': '1',
    'open': '0',
}


class OpenDSSGTCP:
    """
    TCP Client for interacting with OpenDSS-G over the DSTCP protocol (default 127.0.0.1:6345).
    """

    def __init__(self, TCP_IP='127.0.0.1', TCP_PORT=6345, BUFFER_SIZE=65536, timeout=10.0):
        self.TCP_IP = TCP_IP
        self.TCP_PORT = TCP_PORT
        self.BUFFER_SIZE = BUFFER_SIZE
        self.timeout = timeout
        self.s = None

    def startTCPConnection(self):
        """Establishes the TCP socket connection to the OpenDSS-G server."""
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(self.timeout)
        self.s.connect((self.TCP_IP, self.TCP_PORT))

    def closeTCPConnection(self):
        """Closes the TCP socket connection."""
        if self.s is not None:
            try:
                self.s.close()
            finally:
                self.s = None

    # --- DSTCP Standard Protocol Functions ---

    def getTimeSeconds(self):
        """Function 01: Request simulation time in seconds."""
        return self.sendTCP(COMMANDS['seconds'])

    def close(self):
        """Close command (&Q;)."""
        return self.sendTCP(COMMANDS['Close'])

    def getTimeDays(self):
        """Function 02: Request simulation time in days."""
        return self.sendTCP(COMMANDS['days'])

    def getVoltsPU(self):
        """Function 03: Request nodes' voltages in p.u."""
        return self.sendTCP(COMMANDS['Vpu'])

    def getVolts(self):
        """Function 04: Request nodes' complex voltages (a + jb)."""
        return self.sendTCP(COMMANDS['V'])

    def getCapsCurrents(self):
        """Function 05: Request capacitors' currents."""
        return self.sendTCP(COMMANDS['CapC'])

    def getCapsVolts(self):
        """Function 06: Request capacitors' voltages."""
        return self.sendTCP(COMMANDS['CapV'])

    def getCapsLosses(self):
        """Function 07: Request capacitors' losses."""
        return self.sendTCP(COMMANDS['CapL'])

    def getCapsPowers(self):
        """Function 08: Request capacitors' powers."""
        return self.sendTCP(COMMANDS['CapP'])

    def getXfrmsCurrents(self):
        """Function 09: Request transformers' currents."""
        return self.sendTCP(COMMANDS['XfrtC'])

    def getXfrmsVolts(self):
        """Function 10: Request transformers' voltages."""
        return self.sendTCP(COMMANDS['XfrtV'])

    def getXfrmsLosses(self):
        """Function 11: Request transformers' losses."""
        return self.sendTCP(COMMANDS['XfrtL'])

    def getXfrmsPowers(self):
        """Function 12: Request transformers' powers."""
        return self.sendTCP(COMMANDS['XfrtP'])

    def getLoadsCurrents(self):
        """Function 13: Request loads' currents."""
        return self.sendTCP(COMMANDS['LoadC'])

    def getLoadsPowers(self):
        """Function 14: Request loads' powers (kW, kvar per phase)."""
        return self.sendTCP(COMMANDS['LoadP'])

    def getLinesCurrents(self):
        """Function 15: Request lines' currents."""
        return self.sendTCP(COMMANDS['LineC'])

    def getLinesPowers(self):
        """Function 16: Request lines' powers (kW, kvar per phase/terminal)."""
        return self.sendTCP(COMMANDS['LineP'])

    def getLinesLosses(self):
        """Function 17: Request lines' losses."""
        return self.sendTCP(COMMANDS['LineL'])

    def setSwitchControl(self, switchName, onORoff):
        """Function 18: Write switch status (1=closed, 0=open)."""
        state_val = ONOFF.get(str(onORoff).lower(), str(onORoff))
        message = formatTCPMessage(COMMANDS['SwitchC'], f"{switchName}={state_val}")
        return self.sendTCP(message)

    def getSwitchesStatus(self):
        """Function 19: Request switch status and names."""
        return self.sendTCP(COMMANDS['Switch'])

    def fixLoadPower(self, loadName, kwValue):
        """Function 1A: Fix Load Power (kW)."""
        message = formatTCPMessage(COMMANDS['FixPower'], f"{loadName}-{kwValue}")
        return self.sendTCP(message)

    def modifyElementProperty(self, fullElementName, propName, propValue):
        """Function 1B: Modify Element property (e.g., 'load.load_1', 'kw', '60')."""
        message = formatTCPMessage(COMMANDS['FixElement'], f"{fullElementName}-{propName}-{propValue}")
        return self.sendTCP(message)

    def getEnergyMeterData(self, meterName):
        """Function 1C: Request readings of specified energy meter."""
        message = formatTCPMessage(COMMANDS['EnergyM'], meterName)
        return self.sendTCP(message)

    def getBranchNodeIncMatrix(self):
        """Function 1D: Request branch-to-node incidence matrix."""
        return self.sendTCP(COMMANDS['Inc_Mat'])

    def getEVStorageNames(self):
        """Function 1E: Request list of storage elements within the circuit."""
        return self.sendTCP(COMMANDS['EVStorage_List'])

    def getEVStorageStatus(self):
        """Function 1F: Request storage elements status (Name, %stored, kWh, State)."""
        return self.sendTCP(COMMANDS['EVStorage_Status'])

    def getSwitchesCurrents(self):
        """Function 20: Request all switch currents."""
        return self.sendTCP(COMMANDS['SW_Currents'])

    def getSwitchesLosses(self):
        """Function 21: Request all switch losses."""
        return self.sendTCP(COMMANDS['SW_Losses'])

    def getSwitchesPowers(self):
        """Function 22: Request all switch powers."""
        return self.sendTCP(COMMANDS['SW_Powers'])

    def stepIn(self):
        """Function 23: Step in command (advances 1 step in Remote Controlled mode)."""
        return self.sendTCP(COMMANDS['Step_in'])

    def endSimulation(self):
        """Function 24: End simulation command."""
        return self.sendTCP(COMMANDS['End_Sim'])

    def sendCommand(self, command):
        """Function 25: Send arbitrary OpenDSS text command."""
        message = formatTCPMessage(COMMANDS['Command'], command)
        return self.sendTCP(message)

    def getMonitorsList(self):
        """Function 26: Request monitors names."""
        return self.sendTCP(COMMANDS['Monitors'])

    def getMonitorData(self, monitorName):
        """Function 27: Request monitor data."""
        message = formatTCPMessage(COMMANDS['Mon_data'], monitorName)
        return self.sendTCP(message)

    def RefreshGUI(self):
        """Function 31: Refresh OpenDSS-G GUI."""
        return self.sendTCP(COMMANDS['RefreshGUI'])

    def getLineNames(self):
        """Function 33: Get the list of lines."""
        return self.sendTCP(COMMANDS['LineNames'])

    def showOverLoads(self):
        """Function 36: Refresh overloads visualization."""
        return self.sendTCP(COMMANDS['ShowOvLoads'])

    def showVoltages(self):
        """Function 37: Refresh voltage profile visualization."""
        return self.sendTCP(COMMANDS['ShowVolt'])

    def showPower(self):
        """Function 38: Refresh power profile visualization."""
        return self.sendTCP(COMMANDS['ShowPower'])

    def clearControlQueue(self):
        """Function 3F: Clear the control queue."""
        return self.sendTCP(COMMANDS['ClearCtrlQueue'])

    def getControlQueue(self):
        """Function 40: Read the control queue."""
        return self.sendTCP(COMMANDS['GetCtrlQueue'])

    def showIsolated(self):
        """Function 41: Refresh isolated zones visualization."""
        return self.sendTCP(COMMANDS['ShowIsolated'])

    def getBusNames(self):
        """Function 42: Get bus names."""
        return self.sendTCP(COMMANDS['BusNames'])

    def getNodeNames(self):
        """Function 43: Get node names."""
        return self.sendTCP(COMMANDS['NodeNames'])

    def getLoadNames(self):
        """Function 44: Get load names."""
        return self.sendTCP(COMMANDS['LoadNames'])

    def getXfrmNames(self):
        """Function 45: Get transformer names."""
        return self.sendTCP(COMMANDS['XfrtNames'])

    def getCapNames(self):
        """Function 46: Get capacitor names."""
        return self.sendTCP(COMMANDS['CapNames'])

    def getStorageNames(self):
        """Function 47: Get storage names."""
        return self.sendTCP(COMMANDS['StorageNames'])

    def getPVNames(self):
        """Function 4A: Get PVSystem names."""
        return self.sendTCP(COMMANDS['PVNames'])

    def getEngineStatus(self):
        """Function 52: Get OpenDSS-G simulation engine status ('1'=ready, '0'=busy)."""
        return self.sendTCP(COMMANDS['EngineStatus'])

    def getEngineProgress(self):
        """Function 53: Get OpenDSS-G simulation engine progress (%)."""
        return self.sendTCP(COMMANDS['EngineProgress'])

    # --- Convenience Element Query Helpers ---

    def getElementVoltage(self, elementName, elementTerminal=1):
        """Selects an element/terminal and queries its Voltages."""
        message = f"select {elementName} {elementTerminal}"
        confirmation = self.sendCommand(message)
        if confirmation and confirmation[0].lower() == 'ok':
            return self.sendCommand('Voltages')
        return []

    def getElementPowers(self, elementName):
        """Selects an element and queries its Powers (kW, kvar per phase)."""
        message = f"select {elementName}"
        confirmation = self.sendCommand(message)
        if confirmation and confirmation[0].lower() == 'ok':
            return self.sendCommand('Powers')
        return []

    def getElementCurrents(self, elementName):
        """Selects an element and queries its Currents (mag, angle per phase)."""
        message = f"select {elementName}"
        confirmation = self.sendCommand(message)
        if confirmation and confirmation[0].lower() == 'ok':
            return self.sendCommand('Currents')
        return []

    def sendTCP(self, message, isByte=False):
        """Sends a framed DSTCP message and reads the complete framed response."""
        if isByte:
            self.s.sendall(message)
        else:
            self.s.sendall(message.encode('utf-8'))

        # Read at least the 8-byte hex length header + payload + ';'
        chunks = []
        received_len = 0
        expected_total = None

        while True:
            chunk = self.s.recv(self.BUFFER_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
            received_len += len(chunk)

            if expected_total is None and received_len >= 8:
                header_bytes = b"".join(chunks)[:8]
                try:
                    msg_len = int(header_bytes.decode('ascii'), 16)
                    expected_total = 8 + msg_len + 1  # 8-hex header + body + ';'
                except ValueError:
                    expected_total = 0

            if expected_total is not None and (expected_total == 0 or received_len >= expected_total):
                break

        raw_str = b"".join(chunks).decode('utf-8', errors='replace')
        cleaned_data = cleanData(raw_str)
        return formatDataList(cleaned_data)


def formatTCPMessage(command, data):
    """Formats a DSTCP command with 4-digit hex length prefix and trailing semicolon."""
    return f"{command}{len(data):04X}{data};"


def cleanData(data):
    """Strips the 8-character hex length header and trailing semicolon."""
    if isinstance(data, bytes):
        data = data.decode('utf-8', errors='replace')
    temp = data[8:-1] if len(data) >= 9 else data
    return temp.rstrip()


def formatDataList(data):
    """Splits cleaned response string into whitespace/tab-delimited tokens."""
    if isinstance(data, bytes):
        data = data.decode('utf-8', errors='replace')
    return data.split()


def parse_float_token(token) -> float:
    """
    Safely parses a float token returned by OpenDSS-G over DSTCP, handling both
    comma decimal separators (e.g., '18,595728') and trailing commas ('-70,').
    """
    if isinstance(token, bytes):
        token = token.decode('utf-8', errors='replace')
    s = str(token).strip()
    if s.endswith(','):
        s = s[:-1].strip()
    s = s.replace(',', '.')
    return float(s)
